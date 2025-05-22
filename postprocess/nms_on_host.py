#!/usr/bin/env python3
"""
Image detection postprocessing module.

This module provides functionality for processing object detection model outputs,
handling the visualization of bounding boxes and labels on input images.
"""

import colorsys
import json
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np


class ImagePostprocessorNmsOnHost:
    """
    A class to handle the postprocessing of object detection outputs.

    This class handles detection filtering, coordinate normalization,
    and visualization of detection results on input images.

    Attributes:
        scales (tuple): Scaling factors (height, width) used for coordinate normalization.
        pads (tuple): Padding values (x, y) used during preprocessing.
        labels (list): List of class labels for detected objects.
        palette_line (list): RGB color palette for drawing bounding boxes.
        palette_text (list): Complementary RGB color palette for drawing text.
    """

    def __init__(
        self, params: Tuple[Tuple[float, float], Tuple[float, float]], configs: str
    ):
        """
        Initialize the object detection postprocessor with scaling and padding information.

        Parameters:
            params (tuple): A tuple containing scaling factors and padding values:
                           ((x_scale, y_scale), (x_pad, y_pad))
            configs (str): Path to the JSON configuration file containing model information
                          and class labels.

        Raises:
            FileNotFoundError: If the specified configuration file doesn't exist.
            ValueError: If there's an error decoding the JSON configuration file or if
                        the input parameters are invalid.
            TypeError: If the input parameters are not of the expected types.
        """
        # Validate input parameters
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError(
                "params must be a tuple of length 2: ((x_scale, y_scale), (x_pad, y_pad))"
            )
        if not isinstance(configs, str):
            raise TypeError("configs must be a string path to the configuration file")

        def generate_palette_and_complement(
            num_classes: int,
        ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
            """
            Generate color palettes for visualization.

            Creates a set of distinct colors for bounding boxes and their complementary
            colors for text overlay to ensure readability.

            Parameters:
                num_classes (int): Number of object classes to generate colors for.

            Returns:
                tuple: (rgb_palette, complement_palette) containing RGB color tuples
                      for lines and text respectively.
            """
            # Generate color for drawing lines
            hsv_colors = [(i / num_classes, 1.0, 1.0) for i in range(num_classes)]
            rgb_palette = [
                tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*hsv))
                for hsv in hsv_colors
            ]

            # Generate complement color for drawing texts over the line color
            complement_palette = []
            for r, g, b in rgb_palette:
                max_val = max(r, g, b)
                min_val = min(r, g, b)
                comp = (
                    max_val + min_val - r,
                    max_val + min_val - g,
                    max_val + min_val - b,
                )
                complement_palette.append(comp)

            return rgb_palette, complement_palette

        # Read object labels and configuration from JSON file
        try:
            with open(configs, "r") as f:
                model_info = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Label file not found at path: {configs}. Please provide the correct path."
            )
        except json.JSONDecodeError:
            raise ValueError("Error decoding the label JSON file.")
        except Exception as e:
            raise ValueError(f"Unexpected error reading configuration file: {str(e)}")

        # Validate model_info structure
        if not isinstance(model_info, list) or len(model_info) < 2:
            raise ValueError("Invalid model configuration format")

        try:
            model_configs = model_info[0]
            if (
                not isinstance(model_configs, dict)
                or "preprocessing" not in model_configs
            ):
                raise ValueError("Invalid model configuration format")

            input_shape = model_configs["preprocessing"]["input_shape"][:2]

            # Extract padding values from params
            self.pads = params[1]
            if not isinstance(self.pads, tuple) or len(self.pads) != 2:
                raise ValueError("Invalid padding values: expected tuple of length 2")

            # Calculate scaling factors based on input shape and provided scale
            scale = params[0]
            if not isinstance(scale, tuple) or len(scale) != 2:
                raise ValueError("Invalid scale values: expected tuple of length 2")

            self.scales = (input_shape[0] * scale[0], input_shape[1] * scale[1])

            # Store class labels
            self.labels = model_info[1]
            if not isinstance(self.labels, dict):
                raise ValueError(
                    "Invalid label format in configuration: expected dictionary mapping class IDs to labels"
                )
        except (IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid configuration structure: {str(e)}")

        # Generate color palettes for visualization
        self.palette_line, self.palette_text = generate_palette_and_complement(
            len(self.labels)
        )

    def draw_detections(
        self, frame: np.ndarray, detections: Dict[str, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Draw bounding boxes for detected objects on the input frame.

        Parameters:
            frame (np.ndarray): The input image frame to draw on.
            detections (dict): Detection results from the model.

        Returns:
            np.ndarray: The frame with drawn bounding boxes.
        """

        def denormalize_detections(
            detections: Dict[str, List[np.ndarray]],
        ) -> List[List[Union[int, float]]]:
            """
            Convert normalized detection coordinates to pixel coordinates.

            Parameters:
                detections (dict): The normalized detection results.

            Returns:
                list: List of denormalized bounding boxes in the format
                     [class_id, y1, x1, y2, x2, confidence].
            """
            # Pre-compute transformation arrays for better performance
            array_factor = np.array([*self.scales, *self.scales], dtype=np.float32)
            array_offset = np.array([*self.pads, *self.pads], dtype=np.float32)

            # Extract outputs from the first (and only) item in detections dictionary
            try:
                _, outputs = list(detections.items())[0]
            except (IndexError, ValueError):
                return []  # Return empty list if detections is empty or malformed

            bounding_boxes = []

            # Process each class's detections
            for class_id, output in enumerate(outputs):
                for decoded_bbox in output:
                    # Apply scaling and offset to convert to pixel coordinates
                    bounding_box = [
                        class_id,
                        *(decoded_bbox[:-1] * array_factor - array_offset),
                        decoded_bbox[-1],
                    ]
                    bounding_boxes.append(bounding_box)

            return bounding_boxes

        # Convert normalized coordinates to pixel coordinates
        detections = denormalize_detections(detections)

        # Draw each detection on the frame
        for detection in detections:
            label_id = detection[0]
            # Confidence score: prob = detection[5] * 100

            # Extract confidence score
            confidence = detection[5]

            # Draw bounding box around detected object using OpenCV
            top_left = (int(detection[2]), int(detection[1]))
            bottom_right = (int(detection[4]), int(detection[3]))
            cv2.rectangle(
                frame,
                top_left,
                bottom_right,
                color=self.palette_line[label_id],
                thickness=4,
            )

            # Label display processing with confidence score
            label = f"{self.labels[str(label_id)]} {confidence:.2f}"
            font_scale = 0.8
            text_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )

            # Calculate coordinates for the label background rectangle inside the bounding box
            bg_top_left = (top_left[0], top_left[1])
            bg_bottom_right = (
                top_left[0] + text_width,
                top_left[1] + text_height + baseline,
            )

            # Draw background rectangle (filled)
            cv2.rectangle(
                frame,
                bg_top_left,
                bg_bottom_right,
                self.palette_line[label_id],
                -1,  # Fill
            )

            # Draw text
            text_org = (top_left[0], top_left[1] + text_height)
            cv2.putText(
                frame,
                label,
                text_org,
                # (top_left[0], top_left[1] - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                self.palette_text[label_id],
                text_thickness,
            )

        return frame

    def postprocess(
        self, frame: np.ndarray, outputs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Main postprocessing function that processes model outputs and visualizes results.

        This function applies all necessary postprocessing steps to the raw model outputs
        and then visualizes the detection results on the input frame.

        Parameters:
            frame (np.ndarray): The input image frame.
            outputs (dict): Model output tensors containing detection results.

        Returns:
            np.ndarray: The processed frame with object detections visualized.
        """
        # Validate inputs
        if frame is None:
            raise ValueError("Input frame cannot be None")

        if not isinstance(outputs, dict):
            raise TypeError("outputs must be a dictionary of model outputs")

        if len(outputs) != 0:
            # Draw object detections on the frame when outputs are available
            frame = self.draw_detections(frame, outputs)

        return frame
