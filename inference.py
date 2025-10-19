#!/usr/bin/env python3
"""
Hailo Inference Pipeline Module

Provides comprehensive implementation for deploying deep learning models on Hailo hardware 
accelerators with support for both synchronous and asynchronous inference operations.
"""

import argparse
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatOrder,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

from postprocess.classification import ImagePostprocessorClassification
from postprocess.nms_on_host import ImagePostprocessorNmsOnHost
from postprocess.palm_detection import ImagePostprocessorPalmDetection

TIMEOUT_MS: int = 10000


class PerformanceProfiler:
    """Profiles execution times across different pipeline stages."""

    def __init__(self) -> None:
        self.checkpoints: Dict[str, List[float]] = defaultdict(list)
        self.last_time: Optional[float] = None
        self.frame_start_time: Optional[float] = None

    def start_frame(self) -> None:
        """Mark the beginning of a frame processing cycle."""
        self.frame_start_time = time.time()
        self.last_time = self.frame_start_time

    def checkpoint(self, name: str) -> None:
        """
        Record a checkpoint with time elapsed since the last checkpoint.

        Args:
            name: Identifier for this checkpoint
        """
        current_time = time.time()
        if self.last_time is not None:
            elapsed = current_time - self.last_time
            self.checkpoints[name].append(elapsed)
        self.last_time = current_time

    def end_frame(self) -> None:
        """Mark the end of frame processing and record total time."""
        if self.frame_start_time is not None:
            total_time = time.time() - self.frame_start_time
            self.checkpoints["total_frame_time"].append(total_time)

    def print_statistics(self) -> None:
        """Print comprehensive statistics for all recorded checkpoints."""
        print("\n" + "=" * 80)
        print("PERFORMANCE PROFILING RESULTS")
        print("=" * 80)

        if not self.checkpoints:
            print("No profiling data collected.")
            return

        print(
            f"{'Checkpoint':<30} {'Count':>8} {'Min(ms)':>12} {'Max(ms)':>12} "
            f"{'Mean(ms)':>12} {'Var(ms²)':>12}"
        )
        print("-" * 80)

        for name, times in sorted(self.checkpoints.items()):
            if len(times) > 0:
                times_ms = np.array(times) * 1000
                min_time = np.min(times_ms)
                max_time = np.max(times_ms)
                mean_time = np.mean(times_ms)
                var_time = np.var(times_ms)

                print(
                    f"{name:<30} {len(times):>8} {min_time:>12.3f} {max_time:>12.3f} "
                    f"{mean_time:>12.3f} {var_time:>12.3f}"
                )

        print("=" * 80)

        if "total_frame_time" in self.checkpoints:
            total_times = self.checkpoints["total_frame_time"]
            avg_frame_time = np.mean(total_times)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            print(f"\nAverage Frame Processing Time: {avg_frame_time * 1000:.3f} ms")
            print(f"Average FPS (from frame time): {avg_fps:.2f}")

        print("=" * 80 + "\n")


class InferPipeline:
    """
    Manages asynchronous and blocking inference pipelines for Hailo models.

    Supports both synchronous and asynchronous execution modes with various
    post-processing capabilities for classification and detection tasks.
    """

    def __init__(
        self,
        net_path: str,
        batch_size: int,
        is_async: bool,
        is_callback: bool,
        is_nms: bool,
        layer_name_u8: List[str],
        layer_name_u16: List[str],
    ) -> None:
        """
        Initialize the inference pipeline.

        Args:
            net_path: Path to the HEF model file
            batch_size: Number of inputs to process in a single batch
            is_async: Whether to use asynchronous inference mode
            is_callback: Whether to use callbacks for async inference
            is_nms: Whether NMS is enabled in the model
            layer_name_u8: Names of layers outputting uint8 tensors
            layer_name_u16: Names of layers outputting uint16 tensors
        """
        self.out_results: Dict[str, np.ndarray] = {}
        self.layer_name_u8 = layer_name_u8
        self.layer_name_u16 = layer_name_u16

        self.configured_infer_model: Optional[Any] = None
        self.bindings: Optional[Any] = None
        self.job: Optional[Any] = None
        self.is_async = is_async
        self.is_callback = is_callback
        self.is_nms = is_nms

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        try:
            self.vdevice: VDevice = VDevice(params)

            if self.is_async:
                self.infer_model = self.vdevice.create_infer_model(net_path)
                self.infer_model.set_batch_size(batch_size)

                for out_name in self.infer_model.output_names:
                    if out_name in self.layer_name_u8:
                        self.infer_model.output(out_name).set_format_type(
                            FormatType.UINT8
                        )
                    elif out_name in self.layer_name_u16:
                        self.infer_model.output(out_name).set_format_type(
                            FormatType.UINT16
                        )
                    else:
                        self.infer_model.output(out_name).set_format_type(
                            FormatType.FLOAT32
                        )

                self.configured_infer_model = self.infer_model.configure()

            else:
                self.hef = HEF(net_path)
                configure_params = ConfigureParams.create_from_hef(
                    hef=self.hef, interface=HailoStreamInterface.PCIe
                )
                self.network_group = (
                    self.vdevice.configure(self.hef, configure_params)
                )[0]

                self.input_vstreams_params = InputVStreamParams.make(
                    self.network_group,
                    format_type=FormatType.UINT8,
                )
                self.output_vstreams_params = OutputVStreamParams.make(
                    self.network_group, format_type=FormatType.FLOAT32
                )

            self.inference: Callable[[List[np.ndarray]], Optional[Dict[str, np.ndarray]]]
            if self.is_async:
                self.inference = lambda dataset: self.infer_async(dataset)
            else:
                self.inference = lambda dataset: self.infer_pipeline(dataset)

        except Exception as e:
            print(f"Error during inference: {e}")

    def close(self) -> None:
        """Clean up allocated resources."""
        if self.configured_infer_model is not None:
            self.configured_infer_model = None

        self.vdevice.release()

    def callback(self, completion_info: Any) -> None:
        """
        Handle completion of asynchronous inference.

        Args:
            completion_info: Object containing completion status and results
        """
        if completion_info.exception:
            print(f"Inference error: {completion_info.exception}")
        else:
            for out_name in self.bindings._output_names:
                if self.is_nms:
                    self.out_results[out_name] = np.array(
                        self.bindings.output(out_name).get_buffer(), dtype=object
                    )
                else:
                    self.out_results[out_name] = self.bindings.output(
                        out_name
                    ).get_buffer()

    def infer_async(self, infer_inputs: List[np.ndarray]) -> None:
        """
        Perform asynchronous inference on input data.

        Args:
            infer_inputs: List of input data arrays for the model

        Returns:
            None (use wait_and_get_output to retrieve results)
        """
        try:
            self.bindings = self.configured_infer_model.create_bindings()

            for in_name, infer_input in zip(self.infer_model.input_names, infer_inputs):
                self.bindings.input(in_name).set_buffer(infer_input)

            for out_name in self.infer_model.output_names:
                out_buffer: np.ndarray
                if out_name in self.layer_name_u8:
                    out_buffer = np.empty(
                        self.infer_model.output(out_name).shape,
                        dtype=np.uint8,
                    )
                elif out_name in self.layer_name_u16:
                    out_buffer = np.empty(
                        self.infer_model.output(out_name).shape,
                        dtype=np.uint16,
                    )
                else:
                    out_buffer = np.empty(
                        self.infer_model.output(out_name).shape,
                        dtype=np.float32,
                    )

                self.bindings.output(out_name).set_buffer(out_buffer)

            self.configured_infer_model.wait_for_async_ready(timeout_ms=TIMEOUT_MS)

            if self.is_callback:
                self.job = self.configured_infer_model.run_async(
                    [self.bindings], partial(self.callback)
                )
            else:
                self.job = self.configured_infer_model.run_async([self.bindings])

        except Exception as e:
            print(f"Error during inference: {e}")

        return None

    def wait_and_get_ouput(self) -> Dict[str, np.ndarray]:
        """
        Wait for asynchronous inference completion and retrieve results.

        Returns:
            Dictionary mapping output layer names to inference results
        """
        infer_results: Dict[str, np.ndarray] = {}

        try:
            self.job.wait(TIMEOUT_MS)

            for index, out_name in enumerate(self.infer_model.output_names):
                if self.is_callback:
                    buffer = (
                        np.array(self.out_results[out_name], dtype=object)
                        if self.is_nms
                        else self.out_results[out_name]
                    )
                else:
                    if self.is_nms:
                        buffer = np.array(
                            self.bindings.output(
                                self.infer_model.output_names[index]
                            ).get_buffer(),
                            dtype=object,
                        )
                    else:
                        buffer = self.bindings.output(
                            self.infer_model.output_names[index]
                        ).get_buffer()

                infer_results[out_name] = buffer

        except Exception as e:
            print(f"Error during inference: {e}")

        return infer_results

    def infer_pipeline(self, infer_inputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform synchronous (blocking) inference on input data.

        Args:
            infer_inputs: List of input data arrays for the model

        Returns:
            Dictionary mapping output layer names to inference results
        """
        infer_results: Dict[str, np.ndarray] = {}

        try:
            input_data: Dict[str, np.ndarray] = {}
            for i, input_vstream_info in enumerate(self.hef.get_input_vstream_infos()):
                input_data[input_vstream_info.name] = infer_inputs[i][np.newaxis, :]

            with InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            ) as infer_pipeline:
                buffer = infer_pipeline.infer(input_data)
                for i, output_vstream_info in enumerate(
                    self.hef.get_output_vstream_infos()
                ):
                    if self.is_nms:
                        infer_results[output_vstream_info.name] = np.array(
                            buffer[output_vstream_info.name][0], dtype=object
                        )
                    else:
                        infer_results[output_vstream_info.name] = buffer[
                            output_vstream_info.name
                        ].squeeze()

        except Exception as e:
            print(f"Error during inference: {e}")

        return infer_results


def format_tensor_info(
    name: str, format: str, order: str, shape: Tuple[int, ...]
) -> str:
    """
    Generate formatted string representation of a tensor.

    Args:
        name: Tensor name
        format: Data format type (e.g., FLOAT32, UINT8)
        order: Memory layout order (e.g., NHWC, NCHW)
        shape: Tensor dimensions (1D, 2D, or 3D tuple)

    Returns:
        Formatted string describing the tensor's properties

    Raises:
        ValueError: If shape is not a tuple of length 1, 2, or 3
    """
    if not isinstance(shape, tuple) or len(shape) not in [1, 2, 3]:
        raise ValueError("Shape must be a tuple of length 1, 2 or 3")

    format = str(format).rsplit(".", 1)[1]
    order = str(order).rsplit(".", 1)[1]

    if len(shape) == 3:
        H, W, C = shape[:3]
        return f"{name} {format}, {order}({H}x{W}x{C})"
    if len(shape) == 2:
        H, W = shape[:2]
        return f"{name} {format}, {order}({H}x{W})"
    else:
        C = shape[0]
        return f"{name} {format}, {order}({C})"


def validate_input_images(images: List[str], vstream_inputs: List[Any]) -> None:
    """
    Validate that the number of input images matches model requirements.

    Args:
        images: List of input images to process
        vstream_inputs: List of input virtual streams required by the model

    Raises:
        ValueError: If there are fewer images than required input streams
    """
    if len(images) < len(vstream_inputs):
        raise ValueError(
            f"The number of input images ({len(images)}) must match the required "
            f"inputs by the model ({len(vstream_inputs)})."
        )


def preprocess_image_from_array(
    image_array: np.ndarray, shape: Union[Tuple[int, int], int]
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize image to target dimensions while maintaining aspect ratio.

    Args:
        image_array: Input image as numpy array
        shape: Target dimensions as (height, width) or single value

    Returns:
        Tuple of (resized_image, scale_factors, padding_values)
    """
    height, width = image_array.shape[:2]
    target_height, target_width = shape if isinstance(shape, tuple) else (shape, shape)

    scale_x = width / target_width
    scale_y = height / target_height

    img_resized = cv2.resize(
        image_array, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
    )

    return img_resized, (scale_x, scale_y), (0, 0)


def preprocess_image_from_array_with_pad(
    image_array: np.ndarray, shape: Union[Tuple[int, int], int]
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize and pad image to target dimensions while maintaining aspect ratio.

    Args:
        image_array: Input image as numpy array (BGR format)
        shape: Target dimensions as (height, width) or single value

    Returns:
        Tuple of (padded_image_rgb, scale_factors, padding_values)
    """
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    height, width = image_array.shape[:2]
    target_height, target_width = shape if isinstance(shape, tuple) else (shape, shape)

    if height >= width:
        h1 = target_height
        w1 = int((target_height / height) * width)
        padh = 0
        padw = int((target_width - w1) / 2)
        scale = height / h1
    else:
        w1 = target_width
        h1 = int((target_width / width) * height)
        padh = int((target_height - h1) / 2)
        padw = 0
        scale = width / w1

    img_resized = cv2.resize(image_array, (w1, h1), interpolation=cv2.INTER_LANCZOS4)

    padh1, padh2 = padh, target_height - h1 - padh
    padw1, padw2 = padw, target_width - w1 - padw

    img_padded = cv2.copyMakeBorder(
        img_resized, padh1, padh2, padw1, padw2, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    pad = (int(padh * scale), int(padw * scale))

    return img_padded, (scale, scale), pad


def format_and_print_vstream_info(
    vstream_infos: List[Any], is_input: bool = True
) -> Iterator[Tuple[str, Tuple[int, ...], Any, Any]]:
    """
    Format and print virtual stream information.

    Args:
        vstream_infos: List of virtual stream objects
        is_input: Whether the streams are inputs (True) or outputs (False)

    Yields:
        Tuple of (name, shape, format_type, order) for each virtual stream
    """
    for i, vstream in enumerate(vstream_infos):
        try:
            name = vstream.name
            fmt_type = vstream.format.type
            order = vstream.format.order
            shape = vstream.shape

            info = f"{'Input' if is_input else 'Output'} #{i} {format_tensor_info(name, fmt_type, order, shape)}"

            print(info)

            yield name, shape, fmt_type, order
        except AttributeError as e:
            print(f"Error processing vstream #{i}: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the inference application.

    Returns:
        Namespace containing parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Python Inference Example")
    parser.add_argument("images", nargs="*", help="Image to infer")
    parser.add_argument(
        "-n",
        "--net",
        default="./hefs/resnet_v1_50.hef",
        help="Path for the HEF model.",
    )
    parser.add_argument(
        "-p",
        "--postprocess",
        type=str,
        choices=["classification", "palm_detection", "nms_on_host"],
        default="classification",
        help="Type of post process",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Model definition with JSON format",
    )
    parser.add_argument(
        "--synchronous",
        action="store_true",
        help="Use synchronous inference instead of asynchronous inference API.",
    )
    parser.add_argument(
        "--callback",
        action="store_true",
        help="Use callback with asynchronous inference, only valid with asynchronous inference.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        required=False,
        help="Number of images in one batch",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to run the inference pipeline.

    Initializes the inference pipeline, processes images or video input,
    applies post-processing, and displays results with performance metrics.
    """
    input_shape: List[Tuple[int, int]] = []
    layer_name_u8: List[str] = []
    layer_name_u16: List[str] = []

    profiler = PerformanceProfiler()

    args = parse_args()

    is_async = not args.synchronous
    is_callback = False if args.synchronous else args.callback
    is_nms = False

    try:
        hef = HEF(args.net)

        vstream_inputs = hef.get_input_vstream_infos()

        if not vstream_inputs:
            raise ValueError("No input streams found in the model.")

        validate_input_images(args.images, vstream_inputs)

        print("VStream infos(inputs):")

        for in_name, in_shape, _, _ in format_and_print_vstream_info(
            vstream_inputs, is_input=True
        ):
            if len(in_shape) < 2:
                raise ValueError(f"Invalid shape for input {in_name}: {in_shape}")
            input_shape.append(in_shape[:2])

        vstream_outputs = hef.get_output_vstream_infos()

        print("VStream infos(outputs):")

        for out_name, _, out_format, out_order in format_and_print_vstream_info(
            vstream_outputs, is_input=False
        ):
            if out_order == FormatOrder.HAILO_NMS_BY_CLASS:
                is_nms = True

        if len(vstream_inputs) != 1:
            raise ValueError(
                "This program only supports models with a single input tensor. Quitting."
            )

        infer = InferPipeline(
            args.net,
            args.batch_size,
            is_async,
            is_callback,
            is_nms,
            layer_name_u8,
            layer_name_u16,
        )

        image_path = Path(args.images[0])

        if not image_path.exists():
            raise FileNotFoundError(f"File does not exist: {image_path}")

        cap = cv2.VideoCapture(str(image_path))

        if not cap.isOpened():
            raise IOError(f"Unable to open file: {image_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        _, scale, pad = preprocess_image_from_array_with_pad(frame, input_shape[0])

        params: Tuple[Tuple[float, float], Tuple[int, int]] = (scale, pad)
        configs: Optional[str] = args.config

        postprocess: Union[
            ImagePostprocessorNmsOnHost,
            ImagePostprocessorPalmDetection,
            ImagePostprocessorClassification,
        ]
        if args.postprocess == "nms_on_host":
            if args.config is None:
                configs = "./configs/yolov8.json"
            postprocess = ImagePostprocessorNmsOnHost(params, configs)
        elif args.postprocess == "palm_detection":
            if args.config is None:
                configs = "./configs/palm_detection_full.json"
            postprocess = ImagePostprocessorPalmDetection(params, configs)
        elif args.postprocess == "classification":
            if args.config is None:
                configs = "./postprocess/class_names_imagenet.json"
            postprocess = ImagePostprocessorClassification(params, configs, top_n=3)
        else:
            raise ValueError(f"Post process {args.postprocess} does not support yet.")

        is_image = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 1
        loop = True
        last_outputs: Dict[str, np.ndarray] = {}
        frame_count = 0

        overall_start_time = time.time()

        print("\nStarting inference loop with profiling enabled...")
        print("Press 'q' to quit (video mode only)\n")

        while loop:
            profiler.start_frame()

            frame_count += 1

            ret, frame = cap.read()
            profiler.checkpoint("1_frame_read")

            if not ret:
                break

            input_frame, scale, pad = preprocess_image_from_array_with_pad(
                frame, input_shape[0]
            )
            inference_dataset = [input_frame]
            profiler.checkpoint("2_preprocessing")

            try:
                outputs = infer.inference(inference_dataset)
                profiler.checkpoint("3_inference_submit")

                if is_image and not args.synchronous:
                    last_outputs = infer.wait_and_get_ouput()
                    profiler.checkpoint("4_inference_wait")
                elif args.synchronous:
                    last_outputs = outputs

                out_frame = frame
                if len(last_outputs) != 0:
                    out_frame = postprocess.postprocess(frame, last_outputs)
                profiler.checkpoint("5_postprocessing")

                if not is_image and not args.synchronous:
                    last_outputs = infer.wait_and_get_ouput()
                    profiler.checkpoint("6_inference_wait")

            except Exception as e:
                print(f"Error during inference: {e}")
                break

            cv2.imshow("Output", out_frame)
            profiler.checkpoint("7_display")

            profiler.end_frame()

            if is_image:
                loop = False
                cv2.waitKey(0)
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    loop = False

    finally:
        if "infer" in locals():
            infer.close()

        if "cap" in locals():
            cap.release()
            cv2.destroyAllWindows()

    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time

    print("\n" + "=" * 80)
    print("BASIC PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Total execution time: {overall_elapsed_time:.6f} seconds")
    print(f"Total frames processed: {frame_count}")

    if overall_elapsed_time > 0:
        overall_fps = frame_count / overall_elapsed_time
        print(f"Overall throughput: {overall_fps:.2f} FPS")

    print("=" * 80)

    profiler.print_statistics()


if __name__ == "__main__":
    main()
