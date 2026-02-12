#!/usr/bin/env python3
"""
Hailo Inference Pipeline Module
"""

import argparse
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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

from inference_utils import DisplayThread, FrameReaderThread, PerformanceProfiler
from postprocess.classification import ImagePostprocessorClassification
from postprocess.nms_on_host import ImagePostprocessorNmsOnHost
from postprocess.palm_detection import ImagePostprocessorPalmDetection

TIMEOUT_MS: int = 10000


# ============================================================================
# Exception Classes
# ============================================================================


class InferenceError(Exception):
    """Base exception for inference-related errors."""

    pass


class InferenceSubmitError(InferenceError):
    """Exception raised when inference submission fails."""

    pass


class InferenceTimeoutError(InferenceError):
    """Exception raised when inference operation times out."""

    pass


class InferenceWaitError(InferenceError):
    """Exception raised when waiting for inference results fails."""

    pass


class InferencePipelineError(InferenceError):
    """Exception raised during synchronous inference pipeline execution."""

    pass


# ============================================================================
# Helper Functions for Exception Detection
# ============================================================================


def is_hailo_timeout_exception(e: Exception) -> bool:
    """
    Check if an exception is a Hailo timeout exception.

    Args:
        e: Exception to check

    Returns:
        True if exception is HailoRTTimeout
    """
    exception_type = type(e).__name__
    exception_str = str(e).lower()
    return (
        exception_type == "HailoRTTimeout"
        or "timeout" in exception_type.lower()
        or "timeout" in exception_str
        or "timed out" in exception_str
    )


def is_hailo_exception(e: Exception) -> bool:
    """
    Check if an exception is a Hailo runtime exception.

    Args:
        e: Exception to check

    Returns:
        True if exception is HailoRTException or derived type
    """
    exception_type = type(e).__name__
    return (
        exception_type.startswith("HailoRT")
        or "hailo" in exception_type.lower()
        or hasattr(e, "__module__")
        and "hailo" in str(getattr(e, "__module__", "")).lower()
    )


# ============================================================================
# InferPipeline Class
# ============================================================================


class InferPipeline:
    """
    Manages asynchronous and blocking inference pipelines for Hailo models.

    Supports both synchronous and asynchronous execution modes with various
    post-processing capabilities for classification and detection tasks.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

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

        Raises:
            RuntimeError: If initialization fails
        """
        # Initialize all attributes first (for safe cleanup)
        self.out_results: Dict[str, np.ndarray] = {}
        self.layer_name_u8 = layer_name_u8
        self.layer_name_u16 = layer_name_u16

        self.configured_infer_model: Optional[Any] = None
        self.bindings: Optional[Any] = None
        self.job: Optional[Any] = None
        self.is_async = is_async
        self.is_callback = is_callback
        self.is_nms = is_nms

        # Initialize device-related attributes as None
        self.vdevice: Optional[VDevice] = None
        self.infer_model: Optional[Any] = None
        self.hef: Optional[HEF] = None
        self.network_group: Optional[Any] = None
        self.input_vstreams_params: Optional[Any] = None
        self.output_vstreams_params: Optional[Any] = None

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        try:
            self.vdevice = VDevice(params)

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

        except Exception as e:
            # Clean up any partially initialized resources
            try:
                if self.vdevice is not None:
                    self.vdevice.release()
            except Exception:
                pass  # Ignore cleanup errors

            print(f"Error during initialize: {e}")
            raise RuntimeError(f"Failed to initialize InferPipeline: {e}") from e

    def inference(self, dataset: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform inference using configured mode (async or sync).

        Args:
            dataset: List of input arrays

        Returns:
            Empty dict for async mode, results for sync mode

        Raises:
            InferenceError: If inference fails
        """
        if self.is_async:
            self.infer_async(dataset)
            return {}
        else:
            return self.infer_pipeline(dataset)

    def close(self) -> None:
        """Clean up allocated resources safely."""
        try:
            if self.configured_infer_model is not None:
                if hasattr(self.configured_infer_model, "release"):
                    self.configured_infer_model.release()
                self.configured_infer_model = None

            if self.network_group is not None:
                self.network_group = None

            if self.vdevice is not None:
                self.vdevice.release()
                self.vdevice = None
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    def callback(self, completion_info: Any) -> None:
        """
        Handle completion of asynchronous inference.

        Args:
            completion_info: Object containing completion status and results
        """
        if completion_info.exception:
            print(f"Inference callback error: {completion_info.exception}")
        else:
            self.out_results.clear()
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

        Raises:
            InferenceSubmitError: If inference submission fails
            InferenceTimeoutError: If waiting for device times out
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not infer_inputs:
            raise ValueError("infer_inputs cannot be empty")

        if len(infer_inputs) != len(self.infer_model.input_names):
            raise ValueError(
                f"Expected {len(self.infer_model.input_names)} inputs, "
                f"got {len(infer_inputs)}"
            )

        try:
            # Create bindings for this inference job
            self.bindings = self.configured_infer_model.create_bindings()

            # Set input buffers
            for in_name, infer_input in zip(self.infer_model.input_names, infer_inputs):
                if infer_input is None:
                    raise ValueError(f"Input '{in_name}' cannot be None")
                self.bindings.input(in_name).set_buffer(infer_input)

            # Allocate and set output buffers
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

            # Wait for async ready with Hailo-specific exception handling
            try:
                self.configured_infer_model.wait_for_async_ready(timeout_ms=TIMEOUT_MS)
            except Exception as e:
                if is_hailo_timeout_exception(e):
                    print(f"Timeout waiting for inference device: {e}")
                    raise InferenceTimeoutError(
                        f"Inference device not ready: timeout after {TIMEOUT_MS}ms"
                    ) from e
                elif is_hailo_exception(e):
                    print(f"Hailo runtime error while waiting for async ready: {e}")
                    raise InferenceSubmitError(f"Hailo device error: {e}") from e
                else:
                    print(f"Unexpected error waiting for async ready: {e}")
                    raise InferenceSubmitError(
                        f"Failed to wait for async ready: {e}"
                    ) from e

            # Submit async inference job
            try:
                if self.is_callback:
                    self.job = self.configured_infer_model.run_async(
                        [self.bindings], partial(self.callback)
                    )
                else:
                    self.job = self.configured_infer_model.run_async([self.bindings])
            except Exception as e:
                if is_hailo_exception(e):
                    print(f"Hailo runtime error during run_async: {e}")
                    raise InferenceSubmitError(
                        f"Failed to submit inference job to Hailo device: {e}"
                    ) from e
                else:
                    print(f"Unexpected error during run_async: {e}")
                    raise InferenceSubmitError(
                        f"Failed to submit inference job: {e}"
                    ) from e

        except (ValueError, InferenceSubmitError, InferenceTimeoutError):
            raise

        except AttributeError as e:
            print(f"Initialization error during inference submission: {e}")
            raise InferenceSubmitError(
                f"Inference pipeline not properly initialized: {e}"
            ) from e

        except Exception as e:
            print(f"Unexpected error during inference submission: {e}")
            raise InferenceSubmitError(
                f"Unexpected error during inference submission: {e}"
            ) from e

    def wait_and_get_output(self) -> Dict[str, np.ndarray]:
        """
        Wait for asynchronous inference completion and retrieve results.

        Returns:
            Dictionary mapping output layer names to inference results

        Raises:
            InferenceWaitError: If waiting for results fails
            InferenceTimeoutError: If waiting times out
            RuntimeError: If called before submitting an inference job
        """
        # Check if there's a job to wait for
        if self.job is None:
            raise RuntimeError(
                "No inference job to wait for. Call infer_async() first."
            )

        infer_results: Dict[str, np.ndarray] = {}

        try:
            # Wait for job completion with Hailo-specific timeout handling
            try:
                self.job.wait(TIMEOUT_MS)
            except Exception as e:
                if is_hailo_timeout_exception(e):
                    print(f"Inference job timeout: {e}")
                    raise InferenceTimeoutError(
                        f"Inference job did not complete within {TIMEOUT_MS}ms"
                    ) from e
                elif is_hailo_exception(e):
                    print(f"Hailo runtime error during job wait: {e}")
                    raise InferenceWaitError(
                        f"Hailo device error while waiting for results: {e}"
                    ) from e
                else:
                    print(f"Unexpected error during job wait: {e}")
                    raise InferenceWaitError(
                        f"Failed to wait for inference completion: {e}"
                    ) from e

            # Retrieve results from all output layers
            for index, out_name in enumerate(self.infer_model.output_names):
                try:
                    if self.is_callback:
                        # Results stored in callback
                        if out_name not in self.out_results:
                            raise InferenceWaitError(
                                f"Output '{out_name}' not found in callback results. "
                                f"Available outputs: {list(self.out_results.keys())}"
                            )
                        buffer = (
                            np.array(self.out_results[out_name], dtype=object)
                            if self.is_nms
                            else self.out_results[out_name]
                        )
                    else:
                        # Results in bindings
                        try:
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
                        except Exception as e:
                            if is_hailo_exception(e):
                                print(
                                    f"Hailo error retrieving buffer for '{out_name}': {e}"
                                )
                                raise InferenceWaitError(
                                    f"Failed to retrieve output buffer '{out_name}' from Hailo device: {e}"
                                ) from e
                            else:
                                raise

                    infer_results[out_name] = buffer

                except InferenceWaitError:
                    raise

                except Exception as e:
                    print(f"Error retrieving output '{out_name}': {e}")
                    raise InferenceWaitError(
                        f"Failed to retrieve output '{out_name}': {e}"
                    ) from e

        except (InferenceWaitError, InferenceTimeoutError):
            raise

        except AttributeError as e:
            print(f"State error while accessing inference results: {e}")
            raise InferenceWaitError(f"Inference pipeline state is invalid: {e}") from e

        except Exception as e:
            print(f"Unexpected error while waiting for inference: {e}")
            raise InferenceWaitError(
                f"Unexpected error retrieving inference results: {e}"
            ) from e

        return infer_results

    def infer_pipeline(self, infer_inputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform synchronous (blocking) inference on input data.

        Args:
            infer_inputs: List of input data arrays for the model

        Returns:
            Dictionary mapping output layer names to inference results

        Raises:
            InferencePipelineError: If synchronous inference fails
            InferenceTimeoutError: If operation times out
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not infer_inputs:
            raise ValueError("infer_inputs cannot be empty")

        infer_results: Dict[str, np.ndarray] = {}

        try:
            # Prepare input data dictionary
            input_data: Dict[str, np.ndarray] = {}
            input_vstream_infos = self.hef.get_input_vstream_infos()

            if len(infer_inputs) != len(input_vstream_infos):
                raise ValueError(
                    f"Expected {len(input_vstream_infos)} inputs, "
                    f"got {len(infer_inputs)}"
                )

            for i, input_vstream_info in enumerate(input_vstream_infos):
                if infer_inputs[i] is None:
                    raise ValueError(f"Input at index {i} cannot be None")

                # Add batch dimension if needed
                input_data[input_vstream_info.name] = infer_inputs[i][np.newaxis, :]

            # Execute synchronous inference pipeline with Hailo exception handling
            try:
                with InferVStreams(
                    self.network_group,
                    self.input_vstreams_params,
                    self.output_vstreams_params,
                ) as infer_pipeline:
                    # Run inference
                    try:
                        buffer = infer_pipeline.infer(input_data)
                    except Exception as e:
                        if is_hailo_timeout_exception(e):
                            print(f"Inference timeout: {e}")
                            raise InferenceTimeoutError(
                                f"Synchronous inference timed out: {e}"
                            ) from e
                        elif is_hailo_exception(e):
                            print(f"Hailo runtime error during inference: {e}")
                            raise InferencePipelineError(
                                f"Hailo device error during inference: {e}"
                            ) from e
                        else:
                            raise

                    # Extract results
                    output_vstream_infos = self.hef.get_output_vstream_infos()
                    for i, output_vstream_info in enumerate(output_vstream_infos):
                        output_name = output_vstream_info.name

                        if output_name not in buffer:
                            available_outputs = list(buffer.keys())
                            raise InferencePipelineError(
                                f"Expected output '{output_name}' not found in results. "
                                f"Available outputs: {available_outputs}"
                            )

                        try:
                            if self.is_nms:
                                infer_results[output_name] = np.array(
                                    buffer[output_name][0], dtype=object
                                )
                            else:
                                infer_results[output_name] = buffer[
                                    output_name
                                ].squeeze()
                        except Exception as e:
                            print(f"Error processing output '{output_name}': {e}")
                            raise InferencePipelineError(
                                f"Failed to process output '{output_name}': {e}"
                            ) from e

            except (InferencePipelineError, InferenceTimeoutError):
                raise

            except Exception as e:
                if is_hailo_exception(e):
                    print(f"Hailo error in inference pipeline context: {e}")
                    raise InferencePipelineError(
                        f"Hailo device error in inference pipeline: {e}"
                    ) from e
                else:
                    raise

        except (ValueError, InferencePipelineError, InferenceTimeoutError):
            raise

        except AttributeError as e:
            print(f"Initialization error during synchronous inference: {e}")
            raise InferencePipelineError(
                f"Inference pipeline not properly initialized: {e}"
            ) from e

        except KeyError as e:
            print(f"Missing expected data during inference: {e}")
            raise InferencePipelineError(
                f"Inference failed due to missing data: {e}"
            ) from e

        except Exception as e:
            print(f"Unexpected error during synchronous inference: {e}")
            raise InferencePipelineError(
                f"Unexpected error during inference: {e}"
            ) from e

        return infer_results


# ============================================================================
# Helper Functions (unchanged from original)
# ============================================================================


def format_tensor_info(
    name: str, format: str, order: str, shape: Tuple[int, ...]
) -> str:
    """Generate formatted string representation of a tensor."""
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
    """Validate that the number of input images matches model requirements."""
    if len(images) < len(vstream_inputs):
        raise ValueError(
            f"The number of input images ({len(images)}) must match the required "
            f"inputs by the model ({len(vstream_inputs)})."
        )


def preprocess_image_from_array_with_pad(
    image_array: np.ndarray, shape: Union[Tuple[int, int], int]
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Resize and pad image to target dimensions while maintaining aspect ratio."""
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

    img_resized = cv2.resize(image_array, (w1, h1), interpolation=cv2.INTER_AREA)

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
    """Format and print virtual stream information."""
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
    """Parse command-line arguments for the inference application."""
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
        help="Custom model definition with JSON format",
    )
    parser.add_argument(
        "-s",
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profile in inference loop and show statistics.",
    )
    parser.add_argument(
        "--trace",
        type=str,
        metavar="FILE",
        help="Export profiling data to Perfetto trace JSON file (e.g., trace.json). Requires --profile to be enabled.",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to run the inference pipeline."""
    input_shape: List[Tuple[int, int]] = []
    layer_name_u8: List[str] = []
    layer_name_u16: List[str] = []

    profiler = PerformanceProfiler()

    args = parse_args()

    # Validate that --trace requires --profile
    if args.trace and not args.profile:
        print("Error: --trace requires --profile to be enabled")
        print("Usage: Add --profile flag when using --trace")
        return

    is_async = not args.synchronous
    is_callback = False if args.synchronous else args.callback
    is_nms = False

    profiling_enabled = args.profile

    # Initialize variables that might be used in finally block
    infer: Optional[InferPipeline] = None
    cap: Optional[cv2.VideoCapture] = None
    display_thread: Optional[DisplayThread] = None
    frame_reader_thread: Optional[FrameReaderThread] = None

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
            if configs is None:
                configs = "./configs/yolov8.json"
            postprocess = ImagePostprocessorNmsOnHost(params, configs)
        elif args.postprocess == "palm_detection":
            if configs is None:
                configs = "./configs/palm_detection_full.json"
            postprocess = ImagePostprocessorPalmDetection(params, configs)
        elif args.postprocess == "classification":
            if configs is None:
                configs = "./configs/class_names_imagenet.json"
            postprocess = ImagePostprocessorClassification(params, configs, top_n=3)
        else:
            raise ValueError(f"Post process {args.postprocess} does not support yet.")

        is_image = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 1

        # Start frame reader thread for video mode
        if not is_image:
            frame_reader_thread = FrameReaderThread(
                video_source=cap,
                max_queue_size=4,
                profiler=profiler if profiling_enabled else None,
            )
            frame_reader_thread.start()
            print("Frame reader thread started")

        # Start display thread for video mode
        if not is_image:
            display_thread = DisplayThread(
                window_name="Output",
                max_queue_size=2,
                profiler=profiler if profiling_enabled else None,
            )
            display_thread.start()
            print("Display thread started")

        loop = True
        last_outputs: Dict[str, np.ndarray] = {}
        frame_count = 0

        overall_start_time = time.time()

        print("\nStarting inference loop...")
        if profiling_enabled:
            print("Profiling enabled.")
        print("Press 'q' to quit (video mode only)\n")

        while loop:
            if profiling_enabled:
                profiler.start_frame()

            frame_count += 1

            # Read frame: use FrameReaderThread for video, direct read for images
            if is_image:
                ret, frame = cap.read()
                if profiling_enabled:
                    profiler.checkpoint("1_frame_read")
            else:
                # Video mode: get frame from reader thread
                if frame_reader_thread is not None:
                    frame = frame_reader_thread.get_frame()
                    if profiling_enabled:
                        profiler.checkpoint("1_frame_read")
                        # Collect timing data from capture thread
                        frame_reader_thread.collect_timing_data()

                    if frame is None:
                        break
                    ret = True
                else:
                    # Fallback if thread not initialized
                    ret, frame = cap.read()
                    if profiling_enabled:
                        profiler.checkpoint("1_frame_read")

            if not ret:
                break

            input_frame, scale, pad = preprocess_image_from_array_with_pad(
                frame, input_shape[0]
            )
            inference_dataset = [input_frame]
            if profiling_enabled:
                profiler.checkpoint("2_preprocessing")

            try:
                outputs = infer.inference(inference_dataset)
                if profiling_enabled:
                    profiler.checkpoint("3_inference_submit")

                if is_image and not args.synchronous:
                    last_outputs = infer.wait_and_get_output()
                    if profiling_enabled:
                        profiler.checkpoint("4_inference_wait")
                elif args.synchronous:
                    last_outputs = outputs

                out_frame = frame
                if len(last_outputs) != 0:
                    out_frame = postprocess.postprocess(frame, last_outputs)
                if profiling_enabled:
                    profiler.checkpoint("5_postprocessing")

                if not is_image and not args.synchronous:
                    last_outputs = infer.wait_and_get_output()
                    if profiling_enabled:
                        profiler.checkpoint("6_inference_wait")

            except InferenceTimeoutError as e:
                print(f"Inference timeout (frame {frame_count}): {e}")
                continue

            except InferenceSubmitError as e:
                print(f"Failed to submit inference (frame {frame_count}): {e}")
                break

            except InferenceWaitError as e:
                print(f"Failed to get inference results (frame {frame_count}): {e}")
                continue

            except InferencePipelineError as e:
                print(f"Synchronous inference failed (frame {frame_count}): {e}")
                break

            # Display handling
            if is_image:
                cv2.imshow("Output", out_frame)
                if profiling_enabled:
                    profiler.checkpoint("7_display")
                loop = False
                cv2.waitKey(0)
            else:
                if display_thread is not None:
                    display_thread.display(out_frame)
                    if profiling_enabled:
                        profiler.checkpoint("7_display_queue")
                        # Collect timing data from display thread
                        display_thread.collect_timing_data()

                    if display_thread.is_quit_requested():
                        loop = False

            if profiling_enabled:
                profiler.end_frame()

    except InferenceError as e:
        print(f"Fatal inference error: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    finally:
        # Stop frame reader thread first (if it was started)
        if frame_reader_thread is not None:
            print("\nStopping frame reader thread...")
            frame_reader_thread.stop()

        # Stop display thread (if it was started)
        if display_thread is not None:
            print("Stopping display thread...")
            display_thread.stop()

        # Close inference pipeline
        if infer is not None:
            try:
                infer.close()
            except Exception as e:
                print(f"Error closing inference pipeline: {e}")

        # Release video capture
        if cap is not None:
            try:
                cap.release()
            except Exception as e:
                print(f"Error releasing video capture: {e}")

        # Destroy all OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error destroying windows: {e}")

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

    if profiling_enabled:
        profiler.print_statistics()

        # Export Perfetto trace if requested
        if args.trace:
            profiler.export_perfetto_trace(args.trace)

        profiler.draw_stacked_time_chart()
        profiler.draw_detailed_timing_chart()


if __name__ == "__main__":
    main()
