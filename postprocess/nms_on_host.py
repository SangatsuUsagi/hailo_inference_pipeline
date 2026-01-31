"""
Non-Maximum Suppression (NMS) Post-Processing Module for Object Detection.

This module provides functionality for post-processing object detection model outputs
on the host device. It handles the visualization of detection results by drawing
bounding boxes, labels, and confidence scores on images. The module converts
normalized model output coordinates to image coordinates and applies appropriate
visual styling for different object classes.
"""

import colorsys
import json
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np


class ImagePostprocessorNmsOnHost:
    """
    A class for post-processing object detection results and visualizing them on images.

    This class handles the post-processing of object detection model outputs,
    including drawing bounding boxes, labels, and confidence scores on images.
    It converts normalized model output coordinates to image coordinates and
    applies appropriate visual styling for different object classes.

    Attributes:
        pads (Tuple[float, float]): Padding values for x and y coordinates.
        scales (Tuple[float, float]): Scaling factors for x and y coordinates.
        labels (Dict[str, str]): Mapping from class IDs to human-readable labels.
        palette_line (List[Tuple[int, int, int]]): RGB color palette for bounding boxes.
        palette_text (List[Tuple[int, int, int]]): RGB color palette for text labels.
    """

    def __init__(
        self, params: Tuple[Tuple[float, float], Tuple[float, float]], configs: str
    ) -> None:
        """
        Initialize the ImagePostprocessorNmsOnHost with scaling parameters and model configuration.

        Args:
            params (Tuple[Tuple[float, float], Tuple[float, float]]): A tuple containing:
                - scale (Tuple[float, float]): Scaling factors for x and y coordinates.
                - pads (Tuple[float, float]): Padding values for x and y coordinates.
            configs (str): Path to the model configuration JSON file.
        """

        def generate_palette_and_complement(
            num_classes: int,
        ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
            """
            Generate color palettes for visualization of different object classes.

            Creates two color palettes:
            1. A primary palette for drawing bounding boxes
            2. A complementary palette for drawing text with good contrast

            Args:
                num_classes (int): Number of object classes to generate colors for.

            Returns:
                Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
                    A tuple containing:
                    - rgb_palette: List of RGB colors for drawing bounding boxes
                    - complement_palette: List of complementary RGB colors for text
            """
            # Generate evenly distributed HSV colors for distinct visualization
            # hsv_colors is a List[Tuple[float, float, float]]
            hsv_colors: List[Tuple[float, float, float]] = [
                (i / num_classes, 1.0, 1.0) for i in range(num_classes)
            ]
            # Convert HSV colors to RGB for OpenCV compatibility
            # rgb_palette is a List[Tuple[int, int, int]]
            rgb_palette: List[Tuple[int, int, int]] = [
                tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*hsv))  # c is a float
                for hsv in hsv_colors  # hsv is a Tuple[float, float, float]
            ]

            # Generate complementary colors for better text visibility
            complement_palette: List[Tuple[int, int, int]] = []
            for r, g, b in rgb_palette:
                max_val: int = max(r, g, b)
                min_val: int = min(r, g, b)
                # Calculate complementary color using the max+min-value formula
                comp: Tuple[int, int, int] = (
                    max_val + min_val - r,
                    max_val + min_val - g,
                    max_val + min_val - b,
                )
                complement_palette.append(comp)

            return rgb_palette, complement_palette

        # Load model configuration from JSON file
        with open(configs, "r") as f:  # f is a TextIO
            model_info: List = json.load(f)

        # Extract model configuration and input shape
        model_configs: Dict = model_info[0]
        input_shape: List[int] = model_configs["preprocessing"]["input_shape"][:2]

        # Store padding values for coordinate denormalization
        self.pads: Tuple[float, float] = params[1]
        # Extract scaling factors
        scale: Tuple[float, float] = params[0]
        # Calculate final scales by multiplying input shape with scale factors
        self.scales: Tuple[float, float] = (
            input_shape[0] * scale[0],
            input_shape[1] * scale[1],
        )
        # Get class labels from model info
        self.labels: Dict[str, str] = model_info[1]

        # Generate color palettes for visualization
        self.palette_line: List[Tuple[int, int, int]]
        self.palette_text: List[Tuple[int, int, int]]
        self.palette_line, self.palette_text = generate_palette_and_complement(
            len(self.labels)
        )

    def draw_detections(
        self, frame: np.ndarray, detections: Dict[str, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Draw detection bounding boxes, labels, and confidence scores on the frame.

        Args:
            frame (np.ndarray): The input image frame to draw on.
            detections (Dict[str, List[np.ndarray]]): Detection results from the model.

        Returns:
            np.ndarray: The frame with detection visualizations drawn on it.
        """

        def denormalize_detections(
            detections: Dict[str, List[np.ndarray]],
        ) -> List[List[Union[int, float]]]:
            """
            Convert normalized detection coordinates to image coordinates.

            Args:
                detections (Dict[str, List[np.ndarray]]): Dictionary of detection results.

            Returns:
                List[List[Union[int, float]]]: List of bounding boxes with denormalized coordinates.
                Each box contains [class_id, y_min, x_min, y_max, x_max, confidence].
            """
            # Create scaling factor array for efficient vectorized computation
            array_factor: np.ndarray = np.array(
                [*self.scales, *self.scales], dtype=np.float32
            )
            # Create offset array for padding compensation
            array_offset: np.ndarray = np.array(
                [*self.pads, *self.pads], dtype=np.float32
            )

            # Extract outputs from the detections dictionary
            _, outputs = list(detections.items())[0]

            bounding_boxes: List[List[Union[int, float]]] = []

            # Process each class's detections
            for class_id, output in enumerate(outputs):
                for decoded_bbox in output:
                    # Convert normalized coordinates to image coordinates:
                    # 1. Multiply by scaling factor to get to original image size
                    # 2. Subtract padding to compensate for letterboxing
                    bounding_box: List[Union[int, float]] = [
                        class_id,
                        *(decoded_bbox[:-1] * array_factor - array_offset),
                        decoded_bbox[-1],
                    ]
                    bounding_boxes.append(bounding_box)

            return bounding_boxes

        # Convert normalized coordinates to image coordinates
        detections = denormalize_detections(detections)

        # Draw each detection on the frame
        for detection in detections:
            # Extract class ID from detection
            label_id: int = int(detection[0])

            # Extract confidence score
            confidence: float = float(detection[5])

            # Extract bounding box coordinates
            top_left: Tuple[int, int] = (int(detection[2]), int(detection[1]))
            bottom_right: Tuple[int, int] = (int(detection[4]), int(detection[3]))

            # Draw bounding box rectangle
            cv2.rectangle(
                frame,
                top_left,
                bottom_right,
                color=self.palette_line[label_id],
                thickness=4,
            )

            # Prepare label text with class name and confidence
            label: str = f"{self.labels[str(label_id)]} {confidence:.2f}"
            font_scale: float = 0.8
            text_thickness: int = 2

            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )

            # Create coordinates for text background rectangle
            bg_top_left: Tuple[int, int] = (top_left[0], top_left[1])
            bg_bottom_right: Tuple[int, int] = (
                top_left[0] + text_width,
                top_left[1] + text_height + baseline,
            )

            # Draw background rectangle for better text visibility
            cv2.rectangle(
                frame,
                bg_top_left,
                bg_bottom_right,
                self.palette_line[label_id],
                -1,  # Filled rectangle
            )

            # Calculate text position and draw the label
            text_org: Tuple[int, int] = (top_left[0], top_left[1] + text_height)
            cv2.putText(
                frame,
                label,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                self.palette_text[
                    label_id
                ],  # Use complementary color for better visibility
                text_thickness,
            )

        return frame

    def postprocess(
        self, frame: np.ndarray, outputs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Post-process model outputs and visualize detections on the frame.

        This is the main entry point for post-processing object detection results.
        It applies visualization if detections are present in the outputs.

        Args:
            frame (np.ndarray): The input image frame to process.
            outputs (Dict[str, np.ndarray]): Model output tensors.

        Returns:
            np.ndarray: The processed frame with detections drawn on it.
        """
        # Only draw detections if outputs are not empty
        if len(outputs) != 0:
            frame = self.draw_detections(frame, outputs)

        return frame
