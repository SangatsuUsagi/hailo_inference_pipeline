import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class ImagePostprocessorClassification:
    """
    A class for postprocessing classification results from an image inference pipeline.

    Attributes:
        top_n (int): Number of top predictions to consider for each output.
        labels (Optional[Dict[str, str]]): Mapping of label indices to label names loaded from a JSON file.
    """

    def __init__(
        self,
        params: Tuple[Tuple[float, float], Tuple[int, int]],
        configs: str,
        top_n: int = 3,
    ):
        """
        Initialize the postprocessor with parameters, label configuration, and top_n predictions.

        Args:
            params (Tuple[Tuple[float, float], Tuple[int, int]]): Unused parameter for initialization.
            configs (str): Path to the JSON file containing label mappings.
            top_n (int): Number of top predictions to consider for each output. Defaults to 3.

        Raises:
            FileNotFoundError: If the label file is not found at the provided path.
            ValueError: If there is an error decoding the JSON file.
        """
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
        Overlay text annotations on the given image.

        Args:
            image (np.ndarray): The input image on which text will be overlaid.
            strings (List[str]): List of text strings to display on the image.

        Returns:
            np.ndarray: The image with text annotations.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 255)  # Red color in BGR
        thickness = 2

        text_width = 0
        text_height = 0

        # Calculate the maximum text width and height for proper placement.
        for text in strings:
            (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width = max(text_width, w)
            text_height = max(text_height, h)

        x_start = image.shape[1] - text_width
        y_start = 10 + text_height // 2

        # Draw each line of text on the image.
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
        Process the inference outputs and annotate the image with the top predictions.

        Args:
            frame (np.ndarray): The input image to annotate.
            outputs (Dict[str, np.ndarray]): Dictionary of inference outputs with keys as output names
                                             and values as arrays of prediction scores.

        Returns:
            np.ndarray: The annotated image.

        Raises:
            ValueError: If the outputs dictionary is empty or if a label is not found for an index.
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

        # Annotate the image with the predictions.
        self.add_text_to_image(frame, display_string)

        return frame
