#!/usr/bin/env python3

import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class ImagePostprocessorClassification:
    """
    A class to handle the addition of text overlays to images, based on model outputs.

    Attributes:
        labels (dict): Dictionary containing label mappings loaded from the JSON file.
        top_n (int): Number of top predictions to display.

    Methods:
        add_text_to_image(image, strings): Adds text strings to an image at specified positions.
        postprocess(frame, outputs): Post-processes model outputs and displays top predictions on image.
    """

    def __init__(
        self,
        params: Tuple[Tuple[float, float], Tuple[int, int]],
        configs: str,
        top_n: int = 3,
    ):
        """Initializes the ImagePostprocessor with default values for font, scale, color, thickness, and top_n."""
        self.top_n = top_n
        self.labels: Optional[Dict[str, str]] = None

        try:
            with open(configs, "r") as f:
                self.labels = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Label file not found at path: {configs}. Please provide the correct path."
            )
        except json.JSONDecodeError:
            raise ValueError("Error decoding the label JSON file.")

    def add_text_to_image(self, image: np.ndarray, strings: List[str]) -> np.ndarray:
        """
        Adds a list of text strings to an image at specified positions.

        Parameters:
            image (numpy.ndarray): The image as a numpy array.
            strings (list of str): A list of text strings to be added to the image.

        Returns:
            numpy.ndarray: An image object with the text overlay applied.

        Raises:
            ValueError: If the image cannot be loaded or is invalid.

        This function right-aligns the text block at the top-right corner of the
        image, maintaining consistent spacing between lines and starting 10 pixels
        down from the top edge.
        """
        # Calculate maximum width and height for all text lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 255)  # Red color in BGR format
        thickness = 2

        text_width = 0
        text_height = 0

        for text in strings:
            (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width = max(text_width, w)
            text_height = max(text_height, h)

        x_start = image.shape[1] - text_width
        y_start = 10 + text_height // 2

        for i, line in enumerate(strings):
            y_position = y_start + i * (text_height + 10)
            cv2.putText(
                image,
                line,
                (x_start, y_position),
                font,
                font_scale,
                color,
                thickness,
            )

        return image

    def postprocess(
        self, frame: np.ndarray, outputs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Post-processes model outputs and generates top predictions for each output.

        Parameters:
            frame (numpy.ndarray): The video frame or image.
            outputs (dict): A dictionary with keys as output names and values as arrays of prediction probabilities.

        Returns:
            numpy.ndarray: The processed frame with text overlay displaying top predictions.

        Raises:
            ValueError: If a label for an index is not found in the JSON file or if outputs is empty.
        """
        if not outputs:
            raise ValueError("Empty outputs dictionary provided to postprocess method.")

        top_n_predictions = {
            k: outputs[k].argsort()[-self.top_n :][::-1] for k in outputs
        }

        display_string = []
        for key, indices in top_n_predictions.items():
            display_string.append(f"Output: {key}")
            for i, index in enumerate(indices):
                label = self.labels.get(str(index))
                if label is None:
                    raise ValueError(
                        f"Label not found for index '{index}' in JSON file."
                    )
                display_string.append(f"#{i + 1}: {label} ({index})")

        self.add_text_to_image(frame, display_string)

        return frame
