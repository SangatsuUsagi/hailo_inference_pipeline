#!/usr/bin/env python3
"""
Hailo Inference Pipeline Module
"""

import argparse
import time
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    NoReturn,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatOrder,
    FormatType,
    HailoRTException,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

# HailoRTTimeout isn't re-exported from the top-level hailo_platform package
# (only HailoRTException is - see hailo_platform/__init__.py's __all__), so it
# must be imported from its defining submodule directly.
from hailo_platform.pyhailort.pyhailort import HailoRTTimeout

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
        True if exception is a HailoRTTimeout
    """
    return isinstance(e, HailoRTTimeout)


def is_hailo_exception(e: Exception) -> bool:
    """
    Check if an exception is a Hailo runtime exception.

    Args:
        e: Exception to check

    Returns:
        True if exception is a HailoRTException or derived type
    """
    return isinstance(e, HailoRTException)


def _raise_as(
    e: Exception,
    hailo_exc: type,
    hailo_print_msg: str,
    hailo_exc_msg: str,
    *,
    timeout_exc: Optional[type] = None,
    timeout_print_msg: str = "",
    timeout_exc_msg: str = "",
    other_exc: Optional[type] = None,
    other_print_msg: str = "",
    other_exc_msg: str = "",
) -> NoReturn:
    """
    Classify a caught Hailo-related exception and raise the mapped domain exception.

    Checks is_hailo_timeout_exception(e) first (only if timeout_exc is given),
    then is_hailo_exception(e). If neither matches, wraps in other_exc if given,
    otherwise re-raises `e` unchanged. Each branch prints its own message before
    raising, matching this module's existing "log then raise" convention.

    Args:
        e: The caught exception to classify.
        hailo_exc: Domain exception type to raise for a Hailo (non-timeout) error.
        hailo_print_msg: Message to print for the Hailo-error branch.
        hailo_exc_msg: Message for the raised hailo_exc.
        timeout_exc: Domain exception type to raise for a Hailo timeout, if checked.
        timeout_print_msg: Message to print for the timeout branch.
        timeout_exc_msg: Message for the raised timeout_exc.
        other_exc: Domain exception type to raise for a non-Hailo error, if wrapping.
        other_print_msg: Message to print for the non-Hailo branch.
        other_exc_msg: Message for the raised other_exc.
    """
    if timeout_exc is not None and is_hailo_timeout_exception(e):
        print(timeout_print_msg)
        raise timeout_exc(timeout_exc_msg) from e

    if is_hailo_exception(e):
        print(hailo_print_msg)
        raise hailo_exc(hailo_exc_msg) from e

    if other_exc is not None:
        print(other_print_msg)
        raise other_exc(other_exc_msg) from e

    raise


# ============================================================================
# Hailo SDK Handle Protocols
#
# hailo_platform ships without type stubs, so every object it returns is
# typed Any by mypy regardless of our own annotations. These Protocols
# describe only the subset of each SDK object's interface that this module
# actually calls, so attribute access on InferPipeline's handles is checked
# instead of silently accepted.
# ============================================================================


class _InferModelOutputHandle(Protocol):
    """Handle returned by InferModel.output(name)."""

    shape: Tuple[int, ...]

    def set_format_type(self, format_type: "FormatType") -> None: ...


class _InferModelHandle(Protocol):
    """hailo_platform InferModel: describes inputs/outputs before configuration."""

    input_names: List[str]
    output_names: List[str]

    def set_batch_size(self, batch_size: int) -> None: ...
    def output(self, name: str) -> _InferModelOutputHandle: ...
    def configure(self) -> "_ConfiguredInferModelHandle": ...


class _BufferHandle(Protocol):
    """Per-input/output buffer handle exposed by Bindings.input()/.output().

    get_buffer() returns a flat ndarray for regular models, but a ragged
    list of per-class detection arrays for models using on-device NMS
    (FormatOrder.HAILO_NMS_BY_CLASS) - see InferenceOutputs below.
    """

    def get_buffer(self) -> Union[np.ndarray, List[np.ndarray]]: ...
    def set_buffer(self, buffer: np.ndarray) -> None: ...


class _BindingsHandle(Protocol):
    """hailo_platform ConfiguredInferModel.Bindings: buffers for one inference job."""

    _output_names: List[str]

    def input(self, name: str) -> _BufferHandle: ...
    def output(self, name: str) -> _BufferHandle: ...


class _AsyncJobHandle(Protocol):
    """hailo_platform AsyncInferJob: handle returned by ConfiguredInferModel.run_async()."""

    def wait(self, timeout_ms: int) -> None: ...


class _ConfiguredInferModelHandle(Protocol):
    """hailo_platform ConfiguredInferModel: the model after InferModel.configure()."""

    def create_bindings(self) -> _BindingsHandle: ...
    def wait_for_async_ready(self, timeout_ms: int) -> None: ...
    def run_async(
        self, bindings: List[_BindingsHandle], callback: Optional[Any] = None
    ) -> _AsyncJobHandle: ...
    def release(self) -> None: ...


class _NetworkGroupHandle(Protocol):
    """hailo_platform ConfiguredNetworkGroup: opaque handle passed to VStreamParams/InferVStreams."""


class _VStreamParamsHandle(Protocol):
    """hailo_platform Input/OutputVStreamParams: opaque handle passed to InferVStreams."""


# A value is a flat ndarray for regular models, or a list of per-class
# detection arrays (one array per class) for models using on-device NMS
# (FormatOrder.HAILO_NMS_BY_CLASS). Which shape a given InferPipeline
# produces is fixed for its lifetime by the `is_nms` flag passed to
# __init__, and is not encoded per-key in this type - see is_nms.
InferenceOutputs = Dict[str, Union[np.ndarray, List[np.ndarray]]]


# ============================================================================
# InferPipeline Class
# ============================================================================


class InferPipeline:
    """
    Manages asynchronous and blocking inference pipelines for Hailo models.

    Supports both synchronous and asynchronous execution modes with various
    post-processing capabilities for classification and detection tasks.
    """

    def __enter__(self) -> "InferPipeline":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
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
        self.out_results: InferenceOutputs = {}
        self.layer_name_u8 = layer_name_u8
        self.layer_name_u16 = layer_name_u16

        self.configured_infer_model: Optional[_ConfiguredInferModelHandle] = None
        self.bindings: Optional[_BindingsHandle] = None
        self.job: Optional[_AsyncJobHandle] = None
        self.is_async = is_async
        self.is_callback = is_callback
        self.is_nms = is_nms

        # Initialize device-related attributes as None
        self.vdevice: Optional[VDevice] = None
        self.infer_model: Optional[_InferModelHandle] = None
        self.hef: Optional[HEF] = None
        self.network_group: Optional[_NetworkGroupHandle] = None
        self.input_vstreams_params: Optional[_VStreamParamsHandle] = None
        self.output_vstreams_params: Optional[_VStreamParamsHandle] = None

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

    def inference(self, dataset: List[np.ndarray]) -> InferenceOutputs:
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
            if self.bindings is None:
                raise RuntimeError(
                    "InferPipeline callback invoked without active bindings"
                )

            self.out_results.clear()
            for out_name in self.bindings._output_names:
                # get_buffer() already returns a List[np.ndarray] for NMS
                # outputs (one array per class) - store it as-is instead of
                # wrapping in np.array(..., dtype=object), which only
                # obscures the real element type without changing behavior.
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

        if self.infer_model is None or self.configured_infer_model is None:
            raise RuntimeError(
                "InferPipeline is not initialized for asynchronous inference"
            )

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
                _raise_as(
                    e,
                    InferenceSubmitError,
                    f"Hailo runtime error while waiting for async ready: {e}",
                    f"Hailo device error: {e}",
                    timeout_exc=InferenceTimeoutError,
                    timeout_print_msg=f"Timeout waiting for inference device: {e}",
                    timeout_exc_msg=(
                        f"Inference device not ready: timeout after {TIMEOUT_MS}ms"
                    ),
                    other_exc=InferenceSubmitError,
                    other_print_msg=f"Unexpected error waiting for async ready: {e}",
                    other_exc_msg=f"Failed to wait for async ready: {e}",
                )

            # Submit async inference job
            try:
                if self.is_callback:
                    self.job = self.configured_infer_model.run_async(
                        [self.bindings], partial(self.callback)
                    )
                else:
                    self.job = self.configured_infer_model.run_async([self.bindings])
            except Exception as e:
                _raise_as(
                    e,
                    InferenceSubmitError,
                    f"Hailo runtime error during run_async: {e}",
                    f"Failed to submit inference job to Hailo device: {e}",
                    other_exc=InferenceSubmitError,
                    other_print_msg=f"Unexpected error during run_async: {e}",
                    other_exc_msg=f"Failed to submit inference job: {e}",
                )

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

    def wait_and_get_output(self) -> InferenceOutputs:
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

        if self.infer_model is None or self.bindings is None:
            raise RuntimeError(
                "InferPipeline is not initialized for asynchronous inference"
            )

        infer_results: InferenceOutputs = {}

        try:
            # Wait for job completion with Hailo-specific timeout handling
            try:
                self.job.wait(TIMEOUT_MS)
            except Exception as e:
                _raise_as(
                    e,
                    InferenceWaitError,
                    f"Hailo runtime error during job wait: {e}",
                    f"Hailo device error while waiting for results: {e}",
                    timeout_exc=InferenceTimeoutError,
                    timeout_print_msg=f"Inference job timeout: {e}",
                    timeout_exc_msg=(
                        f"Inference job did not complete within {TIMEOUT_MS}ms"
                    ),
                    other_exc=InferenceWaitError,
                    other_print_msg=f"Unexpected error during job wait: {e}",
                    other_exc_msg=f"Failed to wait for inference completion: {e}",
                )

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
                        buffer = self.out_results[out_name]
                    else:
                        # Results in bindings. get_buffer() already returns a
                        # List[np.ndarray] for NMS outputs (one array per
                        # class); no dtype=object wrapping needed.
                        try:
                            buffer = self.bindings.output(
                                self.infer_model.output_names[index]
                            ).get_buffer()
                        except Exception as e:
                            _raise_as(
                                e,
                                InferenceWaitError,
                                f"Hailo error retrieving buffer for '{out_name}': {e}",
                                f"Failed to retrieve output buffer '{out_name}' "
                                f"from Hailo device: {e}",
                            )

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

    def infer_pipeline(self, infer_inputs: List[np.ndarray]) -> InferenceOutputs:
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

        if (
            self.hef is None
            or self.network_group is None
            or self.input_vstreams_params is None
            or self.output_vstreams_params is None
        ):
            raise RuntimeError(
                "InferPipeline is not initialized for synchronous inference"
            )

        infer_results: InferenceOutputs = {}

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
                        _raise_as(
                            e,
                            InferencePipelineError,
                            f"Hailo runtime error during inference: {e}",
                            f"Hailo device error during inference: {e}",
                            timeout_exc=InferenceTimeoutError,
                            timeout_print_msg=f"Inference timeout: {e}",
                            timeout_exc_msg=f"Synchronous inference timed out: {e}",
                        )

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
                                # buffer[output_name][0] is already a
                                # List[np.ndarray] (one array per class);
                                # no dtype=object wrapping needed.
                                infer_results[output_name] = buffer[output_name][0]
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
                _raise_as(
                    e,
                    InferencePipelineError,
                    f"Hailo error in inference pipeline context: {e}",
                    f"Hailo device error in inference pipeline: {e}",
                )

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


class Postprocessor(Protocol):
    """Structural interface satisfied by every postprocessor class selected by --postprocess.

    Replaces a hand-maintained Union[NmsOnHost, PalmDetection, Classification]:
    any class defining a compatible postprocess() method satisfies this
    automatically, without updating this file when a new postprocessor is added.

    `outputs` is intentionally typed Any here rather than InferenceOutputs:
    which concrete shape a given key holds (flat ndarray vs. per-class list)
    depends on the model's is_nms flag, which is fixed per InferPipeline
    instance, not per postprocessor - each concrete postprocess() below
    documents the specific shape it actually expects.
    """

    def postprocess(self, frame: np.ndarray, outputs: Any) -> np.ndarray: ...


def _load_model_and_pipeline(
    args: argparse.Namespace, is_async: bool, is_callback: bool
) -> Tuple[InferPipeline, List[Tuple[int, int]]]:
    """Load the HEF, validate vstream shapes, and construct the configured InferPipeline."""
    input_shape: List[Tuple[int, int]] = []
    layer_name_u8: List[str] = []
    layer_name_u16: List[str] = []
    is_nms = False

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

    return infer, input_shape


def _open_video_source(
    args: argparse.Namespace, input_shape: List[Tuple[int, int]]
) -> Tuple[cv2.VideoCapture, bool, Tuple[Tuple[float, float], Tuple[int, int]]]:
    """Open the image/video source and compute the initial scale/pad for the model's input shape."""
    image_path = Path(args.images[0])

    if not image_path.exists():
        raise FileNotFoundError(f"File does not exist: {image_path}")

    cap = cv2.VideoCapture(str(image_path))

    if not cap.isOpened():
        raise IOError(f"Unable to open file: {image_path}")

    is_image = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 1

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    _, scale, pad = preprocess_image_from_array_with_pad(frame, input_shape[0])

    return cap, is_image, (scale, pad)


def _build_postprocessor(
    args: argparse.Namespace,
    params: Tuple[Tuple[float, float], Tuple[int, int]],
    is_nms: bool,
) -> Postprocessor:
    """Construct the postprocessor selected by --postprocess, applying its default config path.

    Validates --postprocess against is_nms (whether the loaded HEF's output
    format is HAILO_NMS_BY_CLASS): nms_on_host expects a List[np.ndarray]-
    per-class output shape, while classification/palm_detection expect a flat
    np.ndarray. Nothing else ties the CLI flag to the HEF's actual output
    format, so without this check a mismatch would only surface as an opaque
    error deep inside postprocess().
    """
    configs: Optional[str] = args.config

    if args.postprocess == "nms_on_host":
        if not is_nms:
            raise ValueError(
                "--postprocess nms_on_host requires a HEF compiled with "
                "on-device NMS (HAILO_NMS_BY_CLASS output format), but the "
                "loaded HEF does not have one."
            )
        if configs is None:
            configs = "./configs/yolov8.json"
        return ImagePostprocessorNmsOnHost(params, configs)
    elif args.postprocess == "palm_detection":
        if is_nms:
            raise ValueError(
                "--postprocess palm_detection expects flat per-layer output, "
                "but the loaded HEF has on-device NMS output enabled."
            )
        if configs is None:
            configs = "./configs/palm_detection_full.json"
        return ImagePostprocessorPalmDetection(params, configs)
    elif args.postprocess == "classification":
        if is_nms:
            raise ValueError(
                "--postprocess classification expects flat per-layer output, "
                "but the loaded HEF has on-device NMS output enabled."
            )
        if configs is None:
            configs = "./configs/class_names_imagenet.json"
        return ImagePostprocessorClassification(params, configs, top_n=3)
    else:
        raise ValueError(f"Post process {args.postprocess} does not support yet.")


def _start_worker_threads(
    cap: cv2.VideoCapture,
    is_image: bool,
    profiler: PerformanceProfiler,
    profiling_enabled: bool,
) -> Tuple[Optional[FrameReaderThread], Optional[DisplayThread]]:
    """Start the background frame-reader and display threads for video mode."""
    frame_reader_thread: Optional[FrameReaderThread] = None
    display_thread: Optional[DisplayThread] = None

    if not is_image:
        frame_reader_thread = FrameReaderThread(
            video_source=cap,
            max_queue_size=4,
            profiler=profiler if profiling_enabled else None,
        )
        frame_reader_thread.start()
        print("Frame reader thread started")

        display_thread = DisplayThread(
            window_name="Output",
            max_queue_size=2,
            profiler=profiler if profiling_enabled else None,
        )
        display_thread.start()
        print("Display thread started")

    return frame_reader_thread, display_thread


def _run_frame_loop(
    args: argparse.Namespace,
    infer: InferPipeline,
    cap: cv2.VideoCapture,
    postprocess: Postprocessor,
    input_shape: List[Tuple[int, int]],
    frame_reader_thread: Optional[FrameReaderThread],
    display_thread: Optional[DisplayThread],
    profiler: PerformanceProfiler,
    profiling_enabled: bool,
    is_image: bool,
) -> int:
    """Run the capture/inference/postprocess/display loop until quit or end of stream.

    Returns the number of frames processed.
    """
    # get_frame() returning None can mean the reader thread genuinely
    # stopped (end of stream or a read error), or just that its queue was
    # empty for one timeout period while the thread is still running. Only
    # the former should end the loop; the latter is retried, bounded so an
    # unresponsive reader thread still can't hang the pipeline forever.
    MAX_CONSECUTIVE_EMPTY_READS = 30

    loop = True
    last_outputs: InferenceOutputs = {}
    frame_count = 0
    consecutive_empty_reads = 0

    print("\nStarting inference loop...")
    if profiling_enabled:
        print("Profiling enabled.")
    print("Press 'q' to quit\n")

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
                    if frame_reader_thread.has_error():
                        print("Frame reader thread stopped (end of stream or read error)")
                        break

                    consecutive_empty_reads += 1
                    if consecutive_empty_reads >= MAX_CONSECUTIVE_EMPTY_READS:
                        print(
                            f"No frame received for {MAX_CONSECUTIVE_EMPTY_READS} "
                            "consecutive reads; frame reader thread appears "
                            "unresponsive."
                        )
                        break

                    continue

                consecutive_empty_reads = 0
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

    return frame_count


def _cleanup(
    frame_reader_thread: Optional[FrameReaderThread],
    display_thread: Optional[DisplayThread],
    infer: Optional[InferPipeline],
    cap: Optional[cv2.VideoCapture],
) -> None:
    """Stop worker threads, release the inference pipeline and video capture, and close OpenCV windows."""
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


def _print_summary(
    is_image: bool,
    overall_elapsed_time: float,
    frame_count: int,
    profiling_enabled: bool,
    profiler: PerformanceProfiler,
    args: argparse.Namespace,
) -> None:
    """Print the basic throughput summary and, if enabled, profiling statistics/charts."""
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

    if profiling_enabled:
        profiler.print_statistics()

        # Export Perfetto trace if requested
        if args.trace:
            profiler.export_perfetto_trace(args.trace)

        profiler.draw_stacked_time_chart()
        profiler.draw_detailed_timing_chart()


def main() -> None:
    """Parse args, build the inference pipeline, run the frame loop, and print a summary."""
    profiler = PerformanceProfiler()
    args = parse_args()

    # Validate that --trace requires --profile
    if args.trace and not args.profile:
        print("Error: --trace requires --profile to be enabled")
        print("Usage: Add --profile flag when using --trace")
        return

    is_async = not args.synchronous
    is_callback = False if args.synchronous else args.callback
    profiling_enabled = args.profile

    # Initialize variables that must survive to the finally/summary blocks
    # even if setup fails partway through (e.g. a KeyboardInterrupt before
    # the frame loop starts).
    infer: Optional[InferPipeline] = None
    cap: Optional[cv2.VideoCapture] = None
    display_thread: Optional[DisplayThread] = None
    frame_reader_thread: Optional[FrameReaderThread] = None
    is_image = True
    frame_count = 0
    overall_start_time = time.time()

    try:
        infer, input_shape = _load_model_and_pipeline(args, is_async, is_callback)
        cap, is_image, params = _open_video_source(args, input_shape)
        postprocess = _build_postprocessor(args, params, infer.is_nms)
        frame_reader_thread, display_thread = _start_worker_threads(
            cap, is_image, profiler, profiling_enabled
        )

        overall_start_time = time.time()
        frame_count = _run_frame_loop(
            args,
            infer,
            cap,
            postprocess,
            input_shape,
            frame_reader_thread,
            display_thread,
            profiler,
            profiling_enabled,
            is_image,
        )

    except InferenceError as e:
        print(f"Fatal inference error: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    finally:
        _cleanup(frame_reader_thread, display_thread, infer, cap)

    overall_elapsed_time = time.time() - overall_start_time

    _print_summary(
        is_image, overall_elapsed_time, frame_count, profiling_enabled, profiler, args
    )


if __name__ == "__main__":
    main()
