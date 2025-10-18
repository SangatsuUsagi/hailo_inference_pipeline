#!/usr/bin/env python3
"""
Hailo Inference Pipeline Module

This module provides a comprehensive implementation for deploying deep learning models
on Hailo hardware accelerators. It supports both synchronous and asynchronous inference
operations on images and video streams with various post-processing capabilities.

The implementation includes:
- Model loading and configuration
- Input preprocessing and formatting
- Asynchronous and synchronous inference pipelines
- Result post-processing for classification and detection tasks
- Support for various output formats and tensor shapes

Usage:
    python3 inference.py [image files] [options]
"""

import argparse
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, TypeVar, Union

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    FormatOrder,
    FormatType,
    HailoSchedulingAlgorithm,
    InferVStreams,
    VDevice,
)

from postprocess.classification import ImagePostprocessorClassification
from postprocess.nms_on_host import ImagePostprocessorNmsOnHost
from postprocess.palm_detection import ImagePostprocessorPalmDetection

# Configuration constants
TIMEOUT_MS = 10000  # Timeout for asynchronous operations in milliseconds


T = TypeVar("T")


class PerformanceProfiler:
    """
    A profiling class to measure and analyze execution times between checkpoints.

    This class helps identify performance bottlenecks by tracking the time spent
    in different stages of the inference pipeline. It collects timing data for
    each checkpoint and provides statistical analysis (min, max, mean, variance).

    Attributes:
        checkpoints (dict): Dictionary storing lists of elapsed times for each checkpoint.
        last_time (float): Timestamp of the last checkpoint for calculating intervals.
        frame_start_time (float): Timestamp when frame processing started.
    """

    def __init__(self):
        """Initialize the profiler with empty checkpoint storage."""
        # Dictionary to store timing data: {checkpoint_name: [time1, time2, ...]}
        self.checkpoints = defaultdict(list)
        self.last_time = None
        self.frame_start_time = None

    def start_frame(self):
        """
        Mark the start of a new frame processing cycle.

        This should be called at the beginning of each frame iteration to establish
        the baseline for measuring subsequent checkpoint intervals.
        """
        self.frame_start_time = time.time()
        self.last_time = self.frame_start_time

    def checkpoint(self, name: str):
        """
        Record a checkpoint and calculate the time elapsed since the last checkpoint.

        Args:
            name (str): Identifier for this checkpoint (e.g., "preprocessing", "inference_wait").

        The elapsed time is stored for later statistical analysis.
        """
        current_time = time.time()
        if self.last_time is not None:
            # Calculate time elapsed since last checkpoint
            elapsed = current_time - self.last_time
            self.checkpoints[name].append(elapsed)
        self.last_time = current_time

    def end_frame(self):
        """
        Mark the end of frame processing and record total frame time.

        This calculates the total time from frame start to frame end and stores it
        under the "total_frame_time" checkpoint.
        """
        if self.frame_start_time is not None:
            total_time = time.time() - self.frame_start_time
            self.checkpoints["total_frame_time"].append(total_time)

    def print_statistics(self):
        """
        Print comprehensive statistics for all recorded checkpoints.

        For each checkpoint, displays:
        - Number of samples collected
        - Minimum time (in milliseconds)
        - Maximum time (in milliseconds)
        - Mean time (in milliseconds)
        - Variance (in milliseconds squared)

        The statistics help identify which pipeline stages are:
        - Consistently slow (high mean)
        - Inconsistent (high variance)
        - Creating bottlenecks (high max)
        """
        print("\n" + "=" * 80)
        print("PERFORMANCE PROFILING RESULTS")
        print("=" * 80)

        if not self.checkpoints:
            print("No profiling data collected.")
            return

        # Print header for the statistics table
        print(
            f"{'Checkpoint':<30} {'Count':>8} {'Min(ms)':>12} {'Max(ms)':>12} {'Mean(ms)':>12} {'Var(ms²)':>12}"
        )
        print("-" * 80)

        # Calculate and display statistics for each checkpoint
        for name, times in sorted(self.checkpoints.items()):
            if len(times) > 0:
                # Convert times to milliseconds for better readability
                times_ms = np.array(times) * 1000
                min_time = np.min(times_ms)
                max_time = np.max(times_ms)
                mean_time = np.mean(times_ms)
                var_time = np.var(times_ms)

                print(
                    f"{name:<30} {len(times):>8} {min_time:>12.3f} {max_time:>12.3f} {mean_time:>12.3f} {var_time:>12.3f}"
                )

        print("=" * 80)

        # Calculate and display overall throughput metrics
        if "total_frame_time" in self.checkpoints:
            total_times = self.checkpoints["total_frame_time"]
            avg_frame_time = np.mean(total_times)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            print(f"\nAverage Frame Processing Time: {avg_frame_time * 1000:.3f} ms")
            print(f"Average FPS (from frame time): {avg_fps:.2f}")

        print("=" * 80 + "\n")


class InferPipeline:
    """
    Class to manage asynchronous and blocking inference pipelines for processing input data
    using deep learning models stored in Hailo Executable Format (HEF).

    This class provides a flexible interface for loading and running inference on various
    models with support for both synchronous and asynchronous execution modes. It handles
    input and output tensor formatting, buffer management, and execution scheduling.

    Attributes:
        out_results (dict): Dictionary to store output results from asynchronous inference.
        layer_name_u8 (list): List of layer names that produce uint8 format tensors as outputs.
        layer_name_u16 (list): List of layer names that produce uint16 format tensors as outputs.
        configured_infer_model: The configured inference model ready for execution.
        bindings: Handles input/output data bindings for the model.
        job: Represents the current asynchronous inference job.
        is_callback (bool): Whether to use callbacks for asynchronous inference.
        is_nms (bool): Whether NMS (Non-Maximum Suppression) is enabled.
        vdevice: Virtual device instance for hardware acceleration.
    """

    def __init__(
        self,
        net_path: str,
        batch_size: int,
        is_callback: bool,
        is_nms: bool,
        layer_name_u8: List[str],
        layer_name_u16: List[str],
    ) -> None:
        """
        Initialize the InferPipeline class with model configuration and execution parameters.

        Args:
            net_path (str): Path to the Hailo Executable Format (HEF) model file.
            batch_size (int): Number of inputs to process in a single batch.
            is_callback (bool): Whether to use callbacks for asynchronous inference results.
            is_nms (bool): Whether NMS (Non-Maximum Suppression) is enabled in the model.
            layer_name_u8 (list): Names of layers outputting uint8 formatted tensors.
            layer_name_u16 (list): Names of layers outputting uint16 formatted tensors.
        """
        self.out_results = {}  # Store inference results
        # Layers that output uint8 format tensors
        self.layer_name_u8 = layer_name_u8
        # Layers that output uint16 format tensors
        self.layer_name_u16 = layer_name_u16

        self.configured_infer_model = None
        self.bindings = None
        self.job = None
        self.is_callback = is_callback
        self.is_nms = is_nms

        # Create VDevice and set the parameters for scheduling algorithm
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        try:
            self.vdevice = VDevice(params)

            # Load the model onto the device
            self.infer_model = self.vdevice.create_infer_model(net_path)

            # Set batch size for inference operations on the loaded model
            self.infer_model.set_batch_size(batch_size)

            for out_name in self.infer_model.output_names:
                # Default output type is float32
                if out_name in self.layer_name_u8:
                    self.infer_model.output(out_name).set_format_type(FormatType.UINT8)
                elif out_name in self.layer_name_u16:
                    self.infer_model.output(out_name).set_format_type(FormatType.UINT16)
                else:
                    self.infer_model.output(out_name).set_format_type(
                        FormatType.FLOAT32
                    )

            self.configured_infer_model = self.infer_model.configure()

        except Exception as e:
            print(f"Error during inference: {e}")

    def close(self) -> None:
        """
        Clean up resources used by the inference pipeline.

        This method ensures proper cleanup of all allocated resources, including
        the configured inference model and the virtual device. It should be called
        when the pipeline is no longer needed to prevent resource leaks.

        The method performs the following operations:
        1. Resets the configured inference model to free associated resources
        2. Releases the virtual device resources

        This is typically used as part of a teardown process in a try-finally block
        or when an object is no longer needed.
        """
        # Reset the configured inference model to free resources
        if self.configured_infer_model is not None:
            self.configured_infer_model = None

        # Release the vdevice resource
        self.vdevice.release()

    def callback(self, completion_info: Any) -> None:
        """
        Callback function to handle the completion of asynchronous inference.

        This function is invoked when an asynchronous inference operation completes.
        It processes the results, handles any exceptions that occurred during inference,
        and stores the output buffers in the out_results dictionary for later retrieval.

        Args:
            completion_info: An object containing information about the completion status,
                             including exceptions if any occurred during inference, and
                             access to output buffers with inference results.
        """
        if completion_info.exception:
            # Handle exceptions that occurred during inference
            print(f"Inference error: {completion_info.exception}")
        else:
            for out_name in self.bindings._output_names:
                # Store results from each output layer into self.out_results
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

        This method configures and starts an asynchronous inference operation on the provided
        input data. It handles the setup of input and output buffers, configures the model,
        and initiates the asynchronous inference job. Results can be retrieved later using
        the wait_and_get_output method.

        Args:
            infer_inputs (list): List of input data arrays for the model.

        Returns:
            None: The method initiates inference but doesn't wait for results. Call
                  wait_and_get_output() to retrieve the results after completion.
        """
        try:
            # Create bindings to manage input/output data buffers
            self.bindings = self.configured_infer_model.create_bindings()

            # Set input buffers using the provided infer inputs
            for in_name, infer_input in zip(self.infer_model.input_names, infer_inputs):
                self.bindings.input(in_name).set_buffer(infer_input)

            # Allocate and set output buffers based on expected data format type
            for out_name in self.infer_model.output_names:
                out_buffer = np.array([])
                # Default output type is float32
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

            # Set timeout for async inference
            self.configured_infer_model.wait_for_async_ready(timeout_ms=TIMEOUT_MS)

            if self.is_callback:
                # Start inference and use callback function for handling results
                self.job = self.configured_infer_model.run_async(
                    [self.bindings], partial(self.callback)
                )
            else:
                # Start inference without callback
                self.job = self.configured_infer_model.run_async([self.bindings])

        except Exception as e:
            print(f"Error during inference: {e}")

        return None

    def wait_and_get_ouput(self) -> Dict[str, np.ndarray]:
        """
        Wait for an asynchronous inference job to complete and collect the output results.

        This method blocks until either the inference job completes or the specified timeout
        is reached. Once completed, it retrieves the inference results from the output buffers
        and returns them as a dictionary keyed by output layer names.

        Returns:
            dict: A dictionary mapping output layer names to their corresponding inference results.
                 Each result is a numpy array with the shape and type specified during configuration.

        Raises:
            Exception: If an error occurs during waiting or result collection, the exception
                      is caught and printed to the console, and an empty dictionary is returned.
        """
        infer_results = {}

        try:
            # Wait for inference to complete with the specified timeout
            self.job.wait(TIMEOUT_MS)

            # Collect the final results after completion
            for index, out_name in enumerate(self.infer_model.output_names):
                if self.is_callback:
                    # For callback mode, results are already in self.out_results
                    buffer = (
                        np.array(self.out_results[out_name], dtype=object)
                        if self.is_nms
                        else self.out_results[out_name]
                    )
                else:
                    # For no-callback mode, manually collect results after waiting
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
        Perform synchronous (blocking) inference on input data using a network group configuration.

        This method executes inference in a blocking manner, waiting for the results before returning.
        It configures virtual streams for inputs and outputs, runs the inference, and collects
        the results immediately.

        Args:
            infer_inputs (list): List of input data arrays for the model.

        Returns:
            dict: A dictionary mapping output layer names to their corresponding inference results.
                 Each result is a numpy array with the shape and type specified during configuration.
        """
        infer_results = {}

        try:
            # Prepare input data by expanding dimensions for each input stream
            input_data = {}
            for i, input_vstream_info in enumerate(self.hef.get_input_vstream_infos()):
                input_data[input_vstream_info.name] = infer_inputs[i][np.newaxis, :]

            # Perform inference on the network group using configured virtual streams
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
                        # Remove batch dimension for regular outputs
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
    Generate a formatted string representation of a tensor with its properties.

    This function creates a human-readable string describing a tensor's properties including
    its name, data format, memory order, and dimensions. The string format varies based on
    the tensor's shape (1D, 2D, or 3D).

    Args:
        name: The name of the tensor.
        format: The data format type (e.g., FLOAT32, UINT8). Only the last segment after
                the last dot in the string representation is used.
        order: The memory layout order (e.g., NHWC, NCHW). Only the last segment after
               the last dot in the string representation is used.
        shape: The shape dimensions of the tensor. Must be a tuple with 1, 2, or 3 elements.

    Returns:
        A formatted string describing the tensor's properties.

    Raises:
        ValueError: If shape is not a tuple with 1, 2, or 3 elements.

    Example:
        >>> format_tensor_info("input", "FormatType.FLOAT32", "FormatOrder.NHWC", (224, 224, 3))
        'input FLOAT32, NHWC(224x224x3)'
    """
    if not isinstance(shape, tuple) or len(shape) not in [1, 2, 3]:
        raise ValueError("Shape must be a tuple of length 1, 2 or 3")

    # Extract the last segment of the format and order strings (after the last dot)
    format = str(format).rsplit(".", 1)[1]
    order = str(order).rsplit(".", 1)[1]

    # Format the string differently based on the tensor shape dimensionality
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
    Validate that the number of input images matches the required model inputs.

    This function checks if there are enough input images to satisfy the model's
    requirements. It compares the length of the provided images list against the
    number of input streams expected by the model.

    Args:
        images: A list of input images to be processed by the model.
        vstream_inputs: A list of input virtual streams required by the model.

    Raises:
        ValueError: If there are fewer images than required input streams.
    """
    # Check if there are fewer images than required by the model
    if len(images) < len(vstream_inputs):
        raise ValueError(
            f"The number of input images ({len(images)}) must match the required inputs by the model ({len(vstream_inputs)})."
        )


def preprocess_image_from_array(
    image_array: np.ndarray, shape: Union[Tuple[int, int], int]
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize images to target dimensions while maintaining aspect ratio.

    This function resizes the input image to match the target dimensions without
    adding padding. The aspect ratio is preserved by scaling proportionally.

    Args:
        image_array: The input image as a numpy array.
        shape: The target dimensions as (height, width) or a single value
               for both height and width.

    Returns:
        A tuple containing:
            - resized_image: The resized image.
            - scale: Scale factors (scale_x, scale_y) between original and target sizes.
            - pad: Padding values (0, 0) as this function does not apply padding.
    """
    # Determine input dimensions
    height, width = image_array.shape[:2]
    target_height, target_width = shape if isinstance(shape, tuple) else (shape, shape)

    # Calculate scale for resizing
    scale_x = width / target_width
    scale_y = height / target_height

    # Resize image using OpenCV's resize method with LANCZOS interpolation for high-quality downsampling
    img_resized = cv2.resize(
        image_array, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
    )

    return img_resized, (scale_x, scale_y), (0, 0)


def preprocess_image_from_array_with_pad(
    image_array: np.ndarray, shape: Union[Tuple[int, int], int]
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize and pad images to target dimensions while maintaining aspect ratio.

    This function resizes the input image to fit within the target dimensions while
    preserving the aspect ratio. It then pads the image with black borders to reach
    the exact target dimensions.

    Args:
        image_array: The input image as a numpy array (BGR format).
        shape: The target dimensions as (height, width) or a single value
               for both height and width.

    Returns:
        A tuple containing:
            - padded_image: The resized and padded image (RGB format).
            - scale: Scale factors (scale_x, scale_y) between original and target sizes.
            - pad: Padding values (pad_height, pad_width) applied to the original image.
    """
    # Convert OpenCV BGR image to RGB
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # Determine input dimensions
    height, width = image_array.shape[:2]
    target_height, target_width = shape if isinstance(shape, tuple) else (shape, shape)

    # Calculate resize dimensions and padding based on aspect ratio
    if height >= width:  # width <= height
        h1 = target_height
        w1 = int((target_height / height) * width)
        padh = 0
        padw = int((target_width - w1) / 2)
        scale = height / h1
    else:  # height < width
        w1 = target_width
        h1 = int((target_width / width) * height)
        padh = int((target_height - h1) / 2)
        padw = 0
        scale = width / w1

    # Resize image using OpenCV's resize method with LANCZOS interpolation for high-quality downsampling
    img_resized = cv2.resize(image_array, (w1, h1), interpolation=cv2.INTER_LANCZOS4)

    # Pad the resized image to the target dimensions
    padh1, padh2 = padh, target_height - h1 - padh
    padw1, padw2 = padw, target_width - w1 - padw

    img_padded = cv2.copyMakeBorder(
        img_resized, padh1, padh2, padw1, padw2, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Calculate padding in original scale
    pad = (int(padh * scale), int(padw * scale))

    return img_padded, (scale, scale), pad


def format_and_print_vstream_info(
    vstream_infos: List[Any], is_input: bool = True
) -> Iterator[Tuple[str, Tuple[int, ...], Any, Any]]:
    """
    Format and print information about virtual stream objects.

    This function iterates through a list of virtual stream objects, extracts their
    properties (name, format, order, shape), formats this information into a readable
    string, and prints it. It also yields the properties for further processing.

    Args:
        vstream_infos: A list of virtual stream objects with attributes like
                      name, format, and shape.
        is_input: Flag indicating if the streams are inputs (True) or outputs (False).
                 Default is True.

    Yields:
        A tuple containing (name, shape, format_type, order) for each virtual stream.

    Example:
        >>> for name, shape, fmt, order in format_and_print_vstream_info(inputs, is_input=True):
        ...     print(f"Processing {name} with shape {shape}")
    """
    # Iterate over each vstream object with an index
    for i, vstream in enumerate(vstream_infos):
        try:
            # Extract attributes from the vstream object
            name = vstream.name
            fmt_type = vstream.format.type
            order = vstream.format.order
            shape = vstream.shape

            # Format the info string based on whether it's input or output
            info = f"{'Input' if is_input else 'Output'} #{i} {format_tensor_info(name, fmt_type, order, shape)}"

            # Print the formatted information
            print(info)

            # Yield a tuple of name, shape, format, and order type for further processing
            yield name, shape, fmt_type, order
        except AttributeError as e:
            print(f"Error processing vstream #{i}: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the inference application.

    This function defines and processes command-line arguments that control the behavior
    of the inference application, including model selection, post-processing type,
    batch size, and synchronous/asynchronous execution mode.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.

    Command-line Arguments:
        images: A list of image file paths to process (positional arguments).
        -n, --net: Path to the HEF model file (default: "./hefs/resnet_v1_50.hef").
        -p, --postprocess: Type of post-processing to apply (choices: classification,
                          palm_detection, nanodet; default: classification).
        -l, --label: Path to the label definition file in JSON format
                     (default: "./postprocess/class_names_imagenet.json").
        --asynchronous: Flag to use asynchronous inference instead of synchronous mode.
        --callback: Flag to use callbacks with asynchronous inference (requires --asynchronous).
        -b, --batch-size: Number of images to process in one batch (default: 1).
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
        "--asynchronous",
        action="store_true",
        help="Use asynchronous inference instead of synchronous inference API.",
    )
    parser.add_argument(
        "--callback",
        action="store_true",
        help="Use callback with asynchronous inference, only valid with --asynchronous.",
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
    Main function to run the inference pipeline on images or video input.

    This function initializes the inference pipeline based on command-line arguments,
    processes either images or video input, and displays the results. It supports both
    synchronous and asynchronous inference modes and handles various post-processing
    options based on the selected model type.

    The function performs the following steps:
    1. Parse command-line arguments
    2. Load and validate the model
    3. Initialize the inference pipeline
    4. Process each frame from the image or video input
    5. Apply post-processing to the inference results
    6. Display the results
    7. Clean up resources

    Performance metrics (execution time and FPS) are calculated and printed at the end.
    """
    # Initialize lists to store input shapes and layer names based on output format
    input_shape = []
    layer_name_u8 = []  # Layers that output uint8 format tensors
    layer_name_u16 = []  # Layers that output uint16 format tensors

    # Parse command-line arguments
    args = parse_args()

    is_callback = args.callback
    is_nms = False

    try:
        # Load the model using the provided network file path
        hef = HEF(args.net)

        # Retrieve input vstream information from the model
        vstream_inputs = hef.get_input_vstream_infos()

        if not vstream_inputs:
            raise ValueError("No input streams found in the model.")

        # Validate the provided images against expected input vstreams
        validate_input_images(args.images, vstream_inputs)

        print("VStream infos(inputs):")

        # Format and print input information for debugging purposes
        for in_name, in_shape, _, _ in format_and_print_vstream_info(
            vstream_inputs, is_input=True
        ):
            if len(in_shape) < 2:
                raise ValueError(f"Invalid shape for input {in_name}: {in_shape}")
            # Store the first two dimensions of each input shape
            input_shape.append(in_shape[:2])

        # Retrieve output vstream information from the model
        vstream_outputs = hef.get_output_vstream_infos()

        print("VStream infos(outputs):")

        # Categorize layers based on their output format and store their names
        for out_name, _, out_format, out_order in format_and_print_vstream_info(
            vstream_outputs, is_input=False
        ):
            # Assume output is only one for HAILO_NMS_BY_CLASS format order
            if out_order == FormatOrder.HAILO_NMS_BY_CLASS:
                is_nms = True

        # Check input tensor is only one here
        if len(vstream_inputs) != 1:
            raise ValueError(
                "This program only supports models with a single input tensor. Quitting."
            )

        # Initialize the inference pipeline with model and processing parameters
        infer = InferPipeline(
            args.net,
            args.batch_size,
            is_callback,
            is_nms,
            layer_name_u8,
            layer_name_u16,
        )

        image_path = Path(args.images[0])

        if not image_path.exists():
            raise FileNotFoundError(f"File does not exist: {image_path}")

        # Open the video or image file for reading
        cap = cv2.VideoCapture(str(image_path))

        if not cap.isOpened():
            raise IOError(f"Unable to open file: {image_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        _, scale, pad = preprocess_image_from_array_with_pad(frame, input_shape[0])

        # Parameters for post-processing
        params = (scale, pad)
        # Configuration for post-processing
        configs = args.config

        # Initialize PostProcess based on the selected post-processing type
        if args.postprocess == "nms_on_host":
            if args.config is None:
                # Default labels for the classes
                configs = "./configs/yolov8.json"
            postprocess = ImagePostprocessorNmsOnHost(params, configs)
        elif args.postprocess == "palm_detection":
            if args.config is None:
                # Default parameters
                configs = "./configs/palm_detection_full.json"
            postprocess = ImagePostprocessorPalmDetection(params, configs)
        elif args.postprocess == "classification":
            if args.config is None:
                # Default labels for the classes
                configs = "./postprocess/class_names_imagenet.json"
            postprocess = ImagePostprocessorClassification(params, configs, top_n=3)
        else:
            raise ValueError(f"Post process {args.postprocess} does not support yet.")

        # Determine if the input is an image or video based on frame count
        is_image = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 1
        loop = True
        last_outputs = []
        frame_count = 0

        # Record the overall start time for throughput calculation
        overall_start_time = time.time()

        print("\nStarting inference loop with profiling enabled...")
        print("Press 'q' to quit (video mode only)\n")

        while loop:
            frame_count += 1

            ret, frame = cap.read()

            if not ret:
                # End of video or error occurred, exit the loop
                break

            # Preprocess the frame by resizing and padding it according to input shape
            input_frame, scale, pad = preprocess_image_from_array_with_pad(
                frame, input_shape[0]
            )
            inference_dataset = [input_frame]  # Prepare dataset for current frame

            try:
                # Perform asynchronous or synchronous inference based on the argument
                outputs = infer.infer_async(inference_dataset)

                if is_image:
                    # Get inference results if image file is input, in asynchronous inference
                    last_outputs = infer.wait_and_get_ouput()

                # Post-process the frame with the inference results
                # In asynchronous inference, processed last frame results while
                # inference with current frame on device
                out_frame = frame
                if len(last_outputs) != 0:
                    out_frame = postprocess.postprocess(frame, last_outputs)

                # In asynchronous inference, wait for the inference on the device
                # and extract the results from the device here
                if not is_image:
                    last_outputs = infer.wait_and_get_ouput()

            except Exception as e:
                print(f"Error during inference: {e}")
                break

            # Display the resulting frame
            cv2.imshow("Output", out_frame)

            if is_image:
                # For images, wait indefinitely until any key is pressed
                loop = False
                cv2.waitKey(0)
            else:
                # For videos or streams, check every 1ms for 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    loop = False

    # Ensure resources are released even if an error occurs
    finally:
        if "infer" in locals():
            infer.close()

        if "cap" in locals():
            cap.release()
            cv2.destroyAllWindows()

    # Calculate overall performance metrics
    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time

    # Print basic performance summary
    if not is_image:
        print("\n" + "=" * 80)
        print("BASIC PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Total execution time: {overall_elapsed_time:.6f} seconds")
        print(f"Total frames processed: {frame_count}")

        if overall_elapsed_time > 0:
            overall_fps = frame_count / overall_elapsed_time
            print(f"Overall throughput: {overall_fps:.2f} FPS")

        print("=" * 80)


if __name__ == "__main__":
    main()
