#!/usr/bin/env python3

import argparse
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

from postprocess import classification, palm_detection

# Configuration constants
TIMEOUT_MS = 10000  # Timeout for asynchronous operations in milliseconds


class InferPipeline:
    """
    Class to manage asynchronous and blocking inference pipelines for processing input data
    using deep learning models stored in Hailo Executable Format (HEF).

    Attributes:
        out_results: Dictionary to store output results from asynchronous inference.
        layer_name_u8: List of layer names that produce float32 format tensors as outputs.
        layer_name_u16: List of layer names that produce uint16 format tensors as outputs.
    """

    def __init__(
        self, net_path, batch_size, is_async, is_callback, layer_name_u8, layer_name_u16
    ):
        """
        Initialize the InferPipeline class.

        Args:
            layer_name_u8 (list): Names of layers outputting float32 formatted tensors.
            layer_name_u16 (list): Names of layers outputting uint16 formatted tensors.
        """
        self.out_results = {}  # Store inference results
        # Layers that output float32 format tensors
        self.layer_name_u8 = layer_name_u8
        # Layers that output uint16 format tensors
        self.layer_name_u16 = layer_name_u16

        self.configured_infer_model = None
        self.bindings = None
        self.job = None
        self.is_async = is_async
        self.is_callback = is_callback

        # Create VDevice and set the parameters for scheduling algorithm
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        try:
            self.vdevice = VDevice(params)

            if self.is_async:
                # Load the model onto the device
                self.infer_model = self.vdevice.create_infer_model(net_path)

                # Set batch size for inference operations on the loaded model
                self.infer_model.set_batch_size(batch_size)

                for out_name in self.infer_model.output_names:
                    # Default output type is float32
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
            else:
                # Load the HEF model into device and configure it
                self.hef = HEF(net_path)

                # Configure network groups
                configure_params = ConfigureParams.create_from_hef(
                    hef=self.hef, interface=HailoStreamInterface.PCIe
                )
                self.network_group = (
                    self.vdevice.configure(self.hef, configure_params)
                )[0]

                # Create input and output virtual stream parameters for inference
                self.input_vstreams_params = InputVStreamParams.make(
                    self.network_group,
                    format_type=FormatType.UINT8,
                )
                self.output_vstreams_params = OutputVStreamParams.make(
                    self.network_group, format_type=FormatType.FLOAT32
                )

            if self.is_async:
                self.inference = lambda dataset: self.infer_async(dataset)
            else:
                self.inference = lambda dataset: self.infer_pipeline(dataset)

        except Exception as e:
            print(f"Error during inference: {e}")

    def close(self):
        """
        Cleans up resources by setting the configured inference model to None
        and releasing the associated vdevice.

        The method ensures that any allocated or referenced resources are
        properly cleaned up. This is typically used as part of a teardown process
        when an object is no longer needed, preventing resource leaks.
        """
        # Reset the configured inference model to free resources
        if self.configured_infer_model is not None:
            self.configured_infer_model = None

        # Release the vdevice resource
        self.vdevice.release()

    def callback(self, completion_info):
        """
        Callback function to handle the completion of inference.

        Args:
            completion_info: Contains information about the completion status, including
                             exceptions if any occurred during inference.
        """
        if completion_info.exception:
            # Handle exceptions that occurred during inference
            print(f"Inference error: {completion_info.exception}")
        else:
            for out_name in self.bindings._output_names:
                # Store results from each output layer into self.out_results
                self.out_results[out_name] = self.bindings.output(out_name).get_buffer()

    def infer_async(self, infer_inputs):
        """
        Perform asynchronous inference on input data.

        Args:
            infer_inputs (list): List of input frames for the model.
            net_path (str): Path to the HEF (Hailo Executable Format) model file.
            batch_size (int): Number of images processed in a single batch. Default is 1.

        Returns:
            infer_results (dict): Dictionary containing the inference results keyed by output layer names.
        """

        try:
            self.configured_infer_model = self.infer_model.configure()

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

    def wait_and_get_ouput(self):
        """
        Waits for an inference job to complete and collects the output results.

        This function waits for a specified timeout period for the inference job
        to finish. Once completed, it retrieves the results from the model's
        output buffers and stores them in a dictionary keyed by output names.

        Returns:
            dict: A dictionary where keys are output names of the model and values
                are the corresponding buffers containing inference results.

        Raises:
            Exception: If an error occurs during the waiting or result collection process,
                    it will be caught and printed to the console.
        """
        infer_results = {}

        try:
            # Wait for inference to complete to retrieve results from self.out_results
            self.job.wait(TIMEOUT_MS)

            # Collect the final results after completion
            for index, out_name in enumerate(self.infer_model.output_names):
                if self.is_callback:
                    buffer = self.out_results[out_name]
                else:
                    # If no callback is used, manually collect results after waiting
                    buffer = self.bindings.output(
                        self.infer_model.output_names[index]
                    ).get_buffer()

                infer_results[out_name] = buffer

        except Exception as e:
            print(f"Error during inference: {e}")

        return infer_results

    def infer_pipeline(self, infer_inputs):
        """
        Perform blocking inference on input data using a network group configuration.

        Args:
            infer_inputs (list): List of input frames for the model.
            net_path (str): Path to the HEF (Hailo Executable Format) model file.
            batch_size (int): Number of images processed in a single batch. Default is 1.

        Returns:
            infer_results (dict): Dictionary containing the inference results keyed by output layer names.
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
                    infer_results[output_vstream_info.name] = buffer[
                        output_vstream_info.name
                    ].squeeze()

        except Exception as e:
            print(f"Error during inference: {e}")

        return infer_results


def format_tensor_info(name, format, order, shape):
    """Returns a string representation of a tensor with specified format and order.

    This function generates a formatted string that describes the structure and
    attributes of a tensor. It is particularly useful for logging or debugging purposes,
    where understanding tensor dimensions and types may be crucial for diagnosing issues
    in data processing pipelines or machine learning workflows.

    The function takes into account different possible shapes of the tensor, supporting:
    - 1D tensors represented simply by their size.
    - 2D tensors described by height and width.
    - 3D tensors extended with channel information.

    Parameters:
        name (str): The identifier or name of the tensor.
        format: An object representing the data format. It is assumed that this can be
                converted to a string, where only the last segment after a dot is used.
        order: An object representing the storage order in memory. Similar to `format`, it
               should have a string representation with meaningful segments post-dot.
        shape (tuple): A tuple indicating the dimensions of the tensor. Only tuples
                       of length 1, 2, or 3 are supported.

    Returns:
        str: A formatted description string that includes the name, format, order, and
             dimensional details of the tensor.

    Raises:
        ValueError: If `shape` is not a tuple with 1, 2, or 3 elements, which ensures
                    only valid configurations are processed.

    Example Use Cases:
    - For diagnostics, when visualizing model architectures where tensor dimensions must be clear.
    - In logging systems to record tensor metadata at various stages of data processing.

    Notes on Implementation:
    The function employs string manipulations to extract relevant segments from `format` and
    `order`. It uses conditional checks to determine how to format the output based on the size
    of `shape`, which allows for flexible handling of different types of tensors.
    """
    if not isinstance(shape, tuple) or len(shape) not in [1, 2, 3]:
        raise ValueError("tensor must be a tuple of length 1, 2 or 3")

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


def validate_input_images(images, vstream_inputs):
    """
    Validates that the number of input images matches the required inputs by the model.

    This function checks if the length of the provided list of images is at least
    as long as the number of required inputs specified in `vstream_inputs`. If there are
    fewer images than required, it raises a ValueError to indicate this mismatch.

    Parameters:
    - images (list): A list of input images that will be used by the model.
    - vstream_inputs (list or other iterable): An object representing the expected number
      of inputs required by the model. This could be an array-like structure where
      each element corresponds to a required input.

    Raises:
    - ValueError: If the number of provided images is less than the number of required
      inputs, indicating that not enough data has been supplied for processing.

    Returns:
    None: The function does not return any value. It raises an exception if validation fails.
    """
    # Check if there are fewer images than required by the model
    if len(images) < len(vstream_inputs):
        raise ValueError(
            "The number of input images must match the required inputs by the model."
        )


def preprocess_image_from_array(image_array, shape):
    """Resize and pad images to be input for detectors.

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such, the image is resized to fit these dimensions while maintaining aspect ratio.

    Parameters:
    - image_array: numpy array representing the image.
    - shape: tuple or integer specifying the target size (height, width).

    Returns:
    - img: Resized image as a numpy array.
    - scale: Scale factor between original and target sizes.
    - pad: Pixels of padding applied in the original image.
    """

    # Determine input dimensions
    height, width = image_array.shape[:2]
    target_height, target_width = shape if isinstance(shape, tuple) else (shape, shape)

    # Calculate scale for resizing
    scale0 = width / target_width
    scale1 = height / target_height

    # Resize image using OpenCV's resize method with LANCZOS interpolation for high-quality downsampling
    img_resized = cv2.resize(
        image_array, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
    )

    return img_resized, (scale0, scale1), (0, 0)


def preprocess_image_from_array_with_pad(image_array, shape):
    """Resize and pad images to be input for detectors.

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such, the image is resized and padded to fit these dimensions while maintaining aspect ratio.

    Parameters:
    - image_array: numpy array representing the image.
    - shape: tuple or integer specifying the target size (height, width).

    Returns:
    - img: Resized and padded image as a numpy array.
    - scale: Scale factor between original and target sizes.
    - pad: Pixels of padding applied in the original image.
    """

    # Convert OpenCV BGR image to RGB
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # Determine input dimensions
    height, width = image_array.shape[:2]
    target_height, target_width = shape if isinstance(shape, tuple) else (shape, shape)

    if height >= width:  # width <= height
        h1 = int(target_height)
        w1 = int((target_height / height) * width)
        padh = 0
        padw = int((target_width - w1) / 2)
        scale = height / h1
    else:  # height < width
        w1 = int(target_width)
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


def format_and_print_vstream_info(vstream_infos, is_input=True):
    """
    Formats and prints information about vstream objects.

    This function iterates over a list of vstream objects, formats their
    details such as name, format type, order, and shape, and prints this
    information. It also yields the name, shape, and format type of each
    vstream object.

    Parameters:
    - vstream_infos (list): A list of vstream objects containing attributes like
                            name, format, and shape.
    - is_input (bool): Flag indicating if the vstreams are inputs or outputs.
                       Default is True (inputs).

    Yields:
    - tuple: A tuple containing the name, shape, and format type of each vstream.

    Returns:
    - None
    """

    # Iterate over each vstream object with an index
    for i, vstream in enumerate(vstream_infos):
        # Extract attributes from the vstream object
        name = vstream.name
        fmt_type = vstream.format.type
        order = vstream.format.order
        shape = vstream.shape

        # Format the info string based on whether it's input or output
        info = f"{'Input' if is_input else 'Output'} #{i} {format_tensor_info(name, fmt_type, order, shape)}"

        # Print the formatted information
        print(info)

        # Yield a tuple of name, shape, and format type for further processing
        yield name, shape, fmt_type


def parse_args():
    """Parse command-line arguments.

    This function is designed to facilitate the parsing of command-line inputs
    necessary for running an asynchronous inference example. It utilizes the `argparse`
    library to define and manage expected input parameters, providing flexibility
    with default values and descriptions for each argument.

    Returns:
        argparse.Namespace: An object containing the parsed arguments as attributes.

    The function defines several command-line arguments:

    - `images`: A list of image files to be processed. This is optional (`nargs="*"`),
      allowing zero or more images to be specified at runtime.

    - `-n`/`--net`: Specifies the path for the HEF model file, with a default value
      of `"./hefs/resnet_v1_50.hef"`. This allows users to provide an alternative model without
      altering source code.

    - `-l`/`--label`: Defines the label definition file in JSON format, with a default
      path of `"./cifar10.json"`, enabling easy integration with standard datasets or custom labels.

    - `-b`/`--batch-size`: Determines the number of images processed in one batch. This
      argument is optional and has a default value of `1`. It supports integer input, allowing
      optimization for different processing loads while maintaining flexibility through its `required=False` setting.

    The function consolidates these inputs into a namespace object, which can be easily accessed
    throughout the application to control behavior dynamically based on user-specified parameters.
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
        choices=["classification", "palm_detection"],
        default="classification",
        help="Type of post process",
    )
    parser.add_argument(
        "-l",
        "--label",
        default="./postprocess/imagenet.json",
        help="Label definition with JSON format",
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


def main():
    """
    This function handles both image and movie files as input using OpenCV.
    If a video file is detected, it loops through frames for inference; if an image file,
    inference is performed once.

    Exception handling ensures graceful error reporting in case of issues during file loading
    or processing. Additionally, resources are managed carefully to avoid leaks.
    """

    # Initialize lists to store input shapes and layer names based on output format
    input_shape = []
    layer_name_u8 = []  # Layers that output float32 format tensors
    layer_name_u16 = []  # Layers that output uint16 format tensors

    # Parse command-line arguments
    args = parse_args()

    is_async = args.asynchronous
    is_callback = False if not args.asynchronous else args.callback

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
        for in_name, in_shape, _ in format_and_print_vstream_info(
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
        for out_name, _, out_format in format_and_print_vstream_info(
            vstream_outputs, is_input=False
        ):
            pass

        # TODO: Check input tensor is only one here
        if len(vstream_inputs) != 1:
            raise ValueError(
                "This program is only supports the model with the single input tensor. Quitting."
            )

        # Initialize the inference pipeline with model and processing parameters
        infer = InferPipeline(
            args.net,
            args.batch_size,
            is_async,
            is_callback,
            layer_name_u8,
            layer_name_u16,
        )

        image_path = Path(args.images[0])

        if not image_path.exists():
            raise FileNotFoundError(f"File does not exist: {image_path}")

        # Open the video or image file for reading
        cap = cv2.VideoCapture(str(image_path))

        if not cap.isOpened():
            print(f"Unable to open file: {image_path}")
            return

        is_image = False if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 1 else True
        loop = True
        last_outputs = []
        frame_count = 0

        start_time = time.time()
        while loop:
            frame_count = frame_count + 1

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
                outputs = infer.inference(inference_dataset)

                if is_image and args.asynchronous:
                    # Get inference results if image file is input, in asynchronous inference
                    last_outputs = infer.wait_and_get_ouput()
                elif not args.asynchronous:
                    # In synchronous inference, just move outputs to last_outputs
                    last_outputs = outputs

                # Post-process the frame with the inference results
                # In asynchronous inference, processed last frame results while
                # inference with current frame on device
                out_frame = frame
                if len(last_outputs) != 0:
                    if args.postprocess == "palm_detection":
                        out_frame = palm_detection.postprocess(
                            frame, last_outputs, scale[0], pad
                        )
                    elif args.postprocess == "classification":
                        out_frame = classification.postprocess(
                            frame, last_outputs, args.label
                        )
                    else:
                        raise ValueError(
                            f"Post process {args.postprocess} does not support yet."
                        )

                # In asynchronous inference, wait for the inference on the device
                # and extract the results from the device here
                if args.asynchronous:
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

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Function took {elapsed_time:.6f} seconds to run.")
    fps = frame_count / elapsed_time
    print(f"Frame count is {frame_count}, ({fps} FPS)")


if __name__ == "__main__":
    main()
