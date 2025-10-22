"""
Palm Detection Post-processing Module

This module provides functionality for post-processing palm detection model outputs,
including visualization of detection results and region of interest (ROI) extraction.
"""

import json
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


class ImagePostprocessorPalmDetection:
    """
    Post-processor for palm detection model outputs.

    This class handles the post-processing of palm detection model outputs,
    including detection box decoding, non-maximum suppression, and visualization.
    It also provides functionality to extract regions of interest (ROIs) for
    subsequent hand landmark detection.
    """

    scale: float
    pad: Tuple[float, float]
    anchors: NDArray[np.float32]
    model_configs: Dict[str, Any]

    def __init__(
        self, params: Tuple[Tuple[float, float], Tuple[float, float]], configs: str
    ) -> None:
        """
        Initialize the palm detection post-processor.

        Args:
            params: A tuple containing scale and padding information.
                   params[0][0]: Scale factor for image preprocessing
                   params[1]: Padding values (height, width)
            configs: Path to the JSON configuration file containing model parameters
                     and anchor generation options.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If there's an error decoding the JSON file.
        """

        def calculate_scale(
            min_scale: float, max_scale: float, stride_index: int, num_strides: int
        ) -> float:
            """
            Calculate the scale for anchor boxes based on stride index.

            Args:
                min_scale: Minimum scale value
                max_scale: Maximum scale value
                stride_index: Current stride index
                num_strides: Total number of strides

            Returns:
                Calculated scale value
            """
            if num_strides == 1:
                return (max_scale + min_scale) * 0.5
            else:
                return min_scale + (max_scale - min_scale) * stride_index / (
                    num_strides - 1.0
                )

        def generate_anchors(options: Dict[str, Any]) -> NDArray[np.float32]:
            """
            Generate anchor boxes based on provided options.

            Args:
                options: Dictionary containing anchor generation parameters

            Returns:
                Numpy array of generated anchors
            """
            strides_size = len(options["strides"])
            assert options["num_layers"] == strides_size

            anchors = []
            layer_id = 0
            while layer_id < strides_size:
                anchor_height = []
                anchor_width = []
                aspect_ratios = []
                scales = []

                # Process layers with the same stride
                last_same_stride_layer = layer_id
                while (last_same_stride_layer < strides_size) and (
                    options["strides"][last_same_stride_layer]
                    == options["strides"][layer_id]
                ):
                    scale = calculate_scale(
                        options["min_scale"],
                        options["max_scale"],
                        last_same_stride_layer,
                        strides_size,
                    )

                    # Special handling for the lowest layer
                    if (
                        last_same_stride_layer == 0
                        and options["reduce_boxes_in_lowest_layer"]
                    ):
                        aspect_ratios.append(1.0)
                        aspect_ratios.append(2.0)
                        aspect_ratios.append(0.5)
                        scales.append(0.1)
                        scales.append(scale)
                        scales.append(scale)
                    else:
                        # Add anchors for all aspect ratios
                        for aspect_ratio in options["aspect_ratios"]:
                            aspect_ratios.append(aspect_ratio)
                            scales.append(scale)

                        # Add an additional anchor if interpolated_scale_aspect_ratio is specified
                        if options["interpolated_scale_aspect_ratio"] > 0.0:
                            scale_next = (
                                1.0
                                if last_same_stride_layer == strides_size - 1
                                else calculate_scale(
                                    options["min_scale"],
                                    options["max_scale"],
                                    last_same_stride_layer + 1,
                                    strides_size,
                                )
                            )
                            scales.append(np.sqrt(scale * scale_next))
                            aspect_ratios.append(
                                options["interpolated_scale_aspect_ratio"]
                            )

                    last_same_stride_layer += 1

                # Calculate anchor dimensions based on scales and aspect ratios
                for i in range(len(aspect_ratios)):
                    ratio_sqrts = np.sqrt(aspect_ratios[i])
                    anchor_height.append(scales[i] / ratio_sqrts)
                    anchor_width.append(scales[i] * ratio_sqrts)

                # Create anchors for the current feature map
                stride = options["strides"][layer_id]
                feature_map_height = int(np.ceil(options["input_size_height"] / stride))
                feature_map_width = int(np.ceil(options["input_size_width"] / stride))

                for y in range(feature_map_height):
                    for x in range(feature_map_width):
                        for anchor_id in range(len(anchor_height)):
                            # Calculate anchor center positions
                            x_center = (
                                x + options["anchor_offset_x"]
                            ) / feature_map_width
                            y_center = (
                                y + options["anchor_offset_y"]
                            ) / feature_map_height

                            new_anchor = [x_center, y_center, 0, 0]
                            if options["fixed_anchor_size"]:
                                new_anchor[2] = 1.0
                                new_anchor[3] = 1.0
                            else:
                                new_anchor[2] = anchor_width[anchor_id]
                                new_anchor[3] = anchor_height[anchor_id]
                            anchors.append(new_anchor)

                layer_id = last_same_stride_layer

            anchors = np.asarray(anchors, dtype=np.float32)
            return anchors

        # Initialize instance variables
        self.scale = params[0][0]
        self.pad = params[1]

        # Load model configuration
        try:
            with open(configs, "r", encoding="utf-8") as f:
                model_info = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found at path: {configs}. Please provide the correct path."
            )
        except json.JSONDecodeError:
            raise ValueError("Error decoding the label JSON file.")

        # Generate anchors based on model configuration
        self.anchors = generate_anchors(model_info[0])
        self.model_configs = model_info[1]

    def _weighted_non_max_suppression(
        self, detections: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        """
        Apply weighted non-maximum suppression to detections.

        This method suppresses overlapping detections by using a weighted average
        of overlapping boxes rather than picking a single box.

        Args:
            detections: Array of detection boxes and scores

        Returns:
            List of filtered detections after non-maximum suppression
        """

        def intersect(
            box_a: NDArray[np.float32], box_b: NDArray[np.float32]
        ) -> NDArray[np.float32]:
            """
            Calculate the intersection area between boxes.

            Args:
                box_a: First set of boxes
                box_b: Second set of boxes

            Returns:
                Array of intersection areas
            """
            A = box_a.shape[0]
            B = box_b.shape[0]
            max_xy = np.minimum(
                np.repeat(np.expand_dims(box_a[:, 2:], axis=1), B, axis=1),
                np.repeat(np.expand_dims(box_b[:, 2:], axis=0), A, axis=0),
            )
            min_xy = np.maximum(
                np.repeat(np.expand_dims(box_a[:, :2], axis=1), B, axis=1),
                np.repeat(np.expand_dims(box_b[:, :2], axis=0), A, axis=0),
            )
            inter = np.clip((max_xy - min_xy), 0, None)
            return inter[:, :, 0] * inter[:, :, 1]

        def jaccard(
            box_a: NDArray[np.float32], box_b: NDArray[np.float32]
        ) -> NDArray[np.float32]:
            """
            Calculate the Jaccard index (IoU) between boxes.

            Args:
                box_a: First set of boxes
                box_b: Second set of boxes

            Returns:
                Array of IoU values
            """
            inter = intersect(box_a, box_b)
            area_a = np.repeat(
                np.expand_dims(
                    (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]), axis=1
                ),
                inter.shape[1],
                axis=1,
            )
            area_b = np.repeat(
                np.expand_dims(
                    (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]), axis=0
                ),
                inter.shape[0],
                axis=0,
            )
            union = area_a + area_b - inter
            return inter / union

        def overlap_similarity(
            box: NDArray[np.float32], other_boxes: NDArray[np.float32]
        ) -> NDArray[np.float32]:
            """
            Calculate IoU between one box and multiple other boxes.

            Args:
                box: Single box
                other_boxes: Multiple boxes

            Returns:
                Array of IoU values
            """
            return jaccard(np.expand_dims(box, axis=0), other_boxes).squeeze(0)

        # Handle empty detections case
        if len(detections) == 0:
            return []

        output_detections: List[NDArray[np.float32]] = []
        # Sort detections by score (highest first)
        remaining = np.argsort(detections[:, self.model_configs["num_coords"]])[::-1]

        # Process detections until none remain
        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Calculate IoU between first box and all other boxes
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # Identify overlapping boxes
            mask = ious > self.model_configs["min_suppression_threshold"]
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Apply weighted average for overlapping boxes
            weighted_detection = detection.copy()
            if len(overlapping) > 1:
                coordinates = detections[
                    overlapping, : self.model_configs["num_coords"]
                ]
                scores = detections[
                    overlapping,
                    self.model_configs["num_coords"] : self.model_configs["num_coords"]
                    + 1,
                ]
                total_score = scores.sum()
                weighted = np.sum(coordinates * scores, axis=0) / total_score
                weighted_detection[: self.model_configs["num_coords"]] = weighted
                weighted_detection[self.model_configs["num_coords"]] = (
                    total_score / len(overlapping)
                )

            output_detections.append(weighted_detection)

        return output_detections

    def denormalize_detections(
        self, detections: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Convert normalized detection coordinates to image coordinates.

        Args:
            detections: Normalized detection boxes and keypoints

        Returns:
            Denormalized detection coordinates
        """
        # Create a copy to avoid modifying the original array
        result = detections.copy()

        # Denormalize box coordinates (ymin, xmin, ymax, xmax)
        result[:, 0] = (
            result[:, 0] * self.scale * self.model_configs["x_scale"] - self.pad[0]
        )
        result[:, 1] = (
            result[:, 1] * self.scale * self.model_configs["x_scale"] - self.pad[1]
        )
        result[:, 2] = (
            result[:, 2] * self.scale * self.model_configs["x_scale"] - self.pad[0]
        )
        result[:, 3] = (
            result[:, 3] * self.scale * self.model_configs["x_scale"] - self.pad[1]
        )

        # Denormalize keypoint coordinates (alternating x, y values)
        result[:, 4::2] = (
            result[:, 4::2] * self.scale * self.model_configs["x_scale"] - self.pad[1]
        )
        result[:, 5::2] = (
            result[:, 5::2] * self.scale * self.model_configs["x_scale"] - self.pad[0]
        )

        return result

    def postprocess_palm_detection(
        self, outputs: Dict[str, NDArray[np.float32]]
    ) -> List[NDArray[np.float32]]:
        """
        Process model outputs to obtain palm detections.

        Args:
            outputs: Dictionary of model output tensors

        Returns:
            List of detected palms with their bounding boxes, keypoints, and scores

        Raises:
            ValueError: If required output tensors are missing or if there's an error in reshaping
        """

        def _decode_boxes(raw_boxes: NDArray[np.float32]) -> NDArray[np.float32]:
            """
            Decode raw box predictions using anchor boxes.

            Args:
                raw_boxes: Raw box predictions from the model

            Returns:
                Decoded boxes with coordinates in normalized image space
            """
            boxes = np.zeros(raw_boxes.shape, dtype=np.float32)

            # Decode center coordinates
            x_center = (
                raw_boxes[..., 0] / self.model_configs["x_scale"] * self.anchors[:, 2]
                + self.anchors[:, 0]
            )
            y_center = (
                raw_boxes[..., 1] / self.model_configs["y_scale"] * self.anchors[:, 3]
                + self.anchors[:, 1]
            )

            # Decode width and height
            w = raw_boxes[..., 2] / self.model_configs["w_scale"] * self.anchors[:, 2]
            h = raw_boxes[..., 3] / self.model_configs["h_scale"] * self.anchors[:, 3]

            # Convert to box coordinates (ymin, xmin, ymax, xmax)
            boxes[..., 0] = y_center - h / 2.0
            boxes[..., 1] = x_center - w / 2.0
            boxes[..., 2] = y_center + h / 2.0
            boxes[..., 3] = x_center + w / 2.0

            # Decode keypoint coordinates
            for k in range(self.model_configs["num_keypoints"]):
                offset = 4 + k * 2
                keypoint_x = (
                    raw_boxes[..., offset]
                    / self.model_configs["x_scale"]
                    * self.anchors[:, 2]
                    + self.anchors[:, 0]
                )
                keypoint_y = (
                    raw_boxes[..., offset + 1]
                    / self.model_configs["y_scale"]
                    * self.anchors[:, 3]
                    + self.anchors[:, 1]
                )
                boxes[..., offset] = keypoint_x
                boxes[..., offset + 1] = keypoint_y

            return boxes

        def _tensors_to_detections(
            raw_box_tensor: NDArray[np.float32], raw_score_tensor: NDArray[np.float32]
        ) -> List[NDArray[np.float32]]:
            """
            Convert raw model outputs to detection boxes.

            Args:
                raw_box_tensor: Tensor with box coordinates and keypoints
                raw_score_tensor: Tensor with detection scores

            Returns:
                List of detection arrays with boxes, keypoints, and scores
            """
            # Decode boxes from raw predictions
            detection_boxes = _decode_boxes(raw_box_tensor)

            # Apply sigmoid to scores and filter by threshold
            thresh = self.model_configs["score_clipping_thresh"]
            clipped_score_tensor = np.clip(raw_score_tensor, -thresh, thresh)
            detection_scores = 1 / (1 + np.exp(-clipped_score_tensor))
            detection_scores = np.squeeze(detection_scores, axis=-1)

            # Filter detections by score threshold
            mask = detection_scores >= self.model_configs["min_score_thresh"]

            # Combine boxes and scores for each batch
            output_detections: List[NDArray[np.float32]] = []
            for i in range(raw_box_tensor.shape[0]):
                boxes = detection_boxes[i, mask[i]]
                scores = detection_scores[i, mask[i]]
                scores = np.expand_dims(scores, axis=-1)
                boxes_scores = np.concatenate((boxes, scores), axis=-1)
                output_detections.append(boxes_scores)

            return output_detections

        # Verify all required output tensors are present
        required_keys = [
            "palm_detection_full/conv29",
            "palm_detection_full/conv34",
            "palm_detection_full/conv30",
            "palm_detection_full/conv35",
        ]
        for key in required_keys:
            if key not in outputs:
                raise ValueError(f"Missing expected output tensor: {key}")

        # Reshape tensors to expected format
        try:
            conv2D_1 = np.reshape(outputs["palm_detection_full/conv29"], (1, 864, 1))
            conv2D_2 = np.reshape(outputs["palm_detection_full/conv34"], (1, 1152, 1))
            conv2D_3 = np.reshape(outputs["palm_detection_full/conv30"], (1, 864, 18))
            conv2D_4 = np.reshape(outputs["palm_detection_full/conv35"], (1, 1152, 18))
        except ValueError as e:
            raise ValueError(f"Error reshaping tensors: {e}")

        # Concatenate tensors for scores and boxes
        out1 = np.concatenate((conv2D_2, conv2D_1), axis=1)  # Scores
        out2 = np.concatenate((conv2D_4, conv2D_3), axis=1)  # Boxes

        # Verify tensor shapes
        assert out1.shape[0] == 1
        assert out1.shape[1] == self.model_configs["num_anchors"]
        assert out1.shape[2] == 1

        assert out2.shape[0] == 1
        assert out2.shape[1] == self.model_configs["num_anchors"]
        assert out2.shape[2] == self.model_configs["num_coords"]

        # Convert tensors to detections
        detections = _tensors_to_detections(out2, out1)

        # Apply non-maximum suppression
        filtered_detections: List[List[NDArray[np.float32]]] = []
        for i in range(len(detections)):
            wnms_detections = self._weighted_non_max_suppression(detections[i])
            if len(wnms_detections) > 0:
                filtered_detections.append(wnms_detections)

        # Extract detections for the first batch (assumes batch size of 1)
        if len(filtered_detections) > 0:
            normalized_detections = np.array(filtered_detections, dtype=np.float32)[0]
        else:
            normalized_detections = []

        return normalized_detections

    def draw_detections(
        self, image: NDArray[np.uint8], filtered_detections: NDArray[np.float32]
    ) -> NDArray[np.uint8]:
        """
        Draw palm detection boxes and keypoints on the input image.

        Args:
            image: Input image to draw on
            filtered_detections: Detected palms with bounding boxes and keypoints

        Returns:
            Image with drawn detections

        Raises:
            TypeError: If the image is not a numpy array
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a numpy ndarray")

        # Create a copy of the image to avoid modifying the original
        result_image = image.copy()

        if len(filtered_detections) > 0:
            # Convert normalized coordinates to image coordinates
            detections = self.denormalize_detections(filtered_detections)

            # Draw each detected palm
            for i in range(detections.shape[0]):
                # Extract bounding box coordinates
                ymin, xmin, ymax, xmax = detections[i, :4]

                # Draw bounding box rectangle
                top_left = (int(xmin), int(ymin))
                bottom_right = (int(xmax), int(ymax))
                cv2.rectangle(
                    result_image, top_left, bottom_right, color=(0, 0, 255), thickness=4
                )

                # Draw keypoints as circles
                n_keypoints = detections.shape[1] // 2 - 2
                for k in range(n_keypoints):
                    kp_x = int(detections[i, 4 + k * 2])
                    kp_y = int(detections[i, 4 + k * 2 + 1])
                    radius = 10

                    cv2.circle(
                        result_image,
                        (kp_x, kp_y),
                        radius=radius,
                        color=(0, 0, 255),
                        thickness=2,
                    )

        return result_image

    def detection2roi(
        self, detection: NDArray[np.float32]
    ) -> Tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        """
        Convert palm detection to region of interest (ROI) parameters.

        This method calculates the center, scale, and rotation angle for the
        hand ROI based on the palm detection.

        Args:
            detection: Palm detection data with bounding box and keypoints

        Returns:
            Tuple containing:
            - xc: x-coordinate of ROI center
            - yc: y-coordinate of ROI center (with offset)
            - scale: Scale factor for the ROI
            - theta: Rotation angle for the ROI
        """
        # Calculate center of the bounding box
        xc = (detection[:, 1] + detection[:, 3]) / 2
        yc = (detection[:, 0] + detection[:, 2]) / 2
        scale = detection[:, 3] - detection[:, 1]  # Width of the bounding box

        # Apply offset and scaling based on model configuration
        yc += self.model_configs["dy"] * scale
        scale *= self.model_configs["dscale"]

        # Calculate rotation angle based on keypoints
        x0 = detection[:, 4 + 2 * self.model_configs["kp1"]]
        y0 = detection[:, 4 + 2 * self.model_configs["kp1"] + 1]
        x1 = detection[:, 4 + 2 * self.model_configs["kp2"]]
        y1 = detection[:, 4 + 2 * self.model_configs["kp2"] + 1]
        theta = np.arctan2(y0 - y1, x0 - x1) - self.model_configs["theta0"]

        return xc, yc, scale, theta

    def extract_roi(
        self,
        xc: NDArray[np.float32],
        yc: NDArray[np.float32],
        theta: NDArray[np.float32],
        scale: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Extract ROI coordinates based on center, scale, and rotation angle.

        Args:
            xc: x-coordinate of ROI center
            yc: y-coordinate of ROI center
            theta: Rotation angle for the ROI
            scale: Scale factor for the ROI

        Returns:
            Array of coordinates for the ROI corners
        """
        # Reshape scale for broadcasting
        scaleN = scale.reshape(-1, 1, 1).astype(np.float32)

        # Define the base rectangle points (4 corners)
        points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=np.float32)

        # Apply scaling
        points = points * scaleN / 2
        points = points.astype(np.float32)

        # Create rotation matrices for each detection
        R = np.zeros((theta.shape[0], 2, 2), dtype=np.float32)

        for i in range(theta.shape[0]):
            R[i, :, :] = [
                [np.cos(theta[i]), -np.sin(theta[i])],
                [np.sin(theta[i]), np.cos(theta[i])],
            ]

        # Combine center coordinates
        center = np.column_stack((xc, yc))
        center = np.expand_dims(center, axis=-1)

        # Apply rotation and translation
        points = np.matmul(R, points) + center
        points = points.astype(np.float32)

        return points

    def draw_roi(
        self, image: NDArray[np.uint8], filtered_detections: NDArray[np.float32]
    ) -> NDArray[np.uint8]:
        """
        Draw region of interest (ROI) on the input image.

        Args:
            image: Input image to draw on
            filtered_detections: Detected palms with bounding boxes and keypoints

        Returns:
            Image with drawn ROI

        Raises:
            TypeError: If the image is not a numpy array
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a numpy ndarray")

        # Create a copy of the image to avoid modifying the original
        result_image = image.copy()

        # Convert detection to ROI parameters
        xc, yc, scale, theta = self.detection2roi(filtered_detections)

        # Extract ROI coordinates
        roi_box = self.extract_roi(xc, yc, theta, scale)

        # Draw each ROI as a quadrilateral
        for i in range(roi_box.shape[0]):
            (x1, x2, x3, x4), (y1, y2, y3, y4) = roi_box[i]

            # Define the four corners of the ROI
            pt1_start = (int(x1), int(y1))
            pt2_mid_top = (int(x2), int(y2))
            pt2_mid_bottom = (int(x3), int(y3))
            pt3_end = (int(x4), int(y4))

            # Draw the ROI as a blue quadrilateral
            cv2.line(
                result_image, pt1_start, pt2_mid_top, color=(255, 0, 0), thickness=3
            )
            cv2.line(
                result_image, pt1_start, pt2_mid_bottom, color=(255, 0, 0), thickness=3
            )
            cv2.line(result_image, pt2_mid_top, pt3_end, color=(255, 0, 0), thickness=3)
            cv2.line(
                result_image, pt2_mid_bottom, pt3_end, color=(255, 0, 0), thickness=3
            )

        return result_image

    def postprocess(
        self, frame: NDArray[np.uint8], outputs: Dict[str, NDArray[np.float32]]
    ) -> NDArray[np.uint8]:
        """
        Main post-processing method that handles palm detection and visualization.

        Args:
            frame: Input image frame
            outputs: Dictionary of model output tensors

        Returns:
            Processed frame with palm detections and ROIs drawn
        """
        # Create a copy of the frame to avoid modifying the original
        result_frame = frame.copy()

        # Process model outputs to get palm detections
        filtered_detections = self.postprocess_palm_detection(outputs)

        # Draw detections and ROIs if any palms were detected
        if len(filtered_detections) != 0:
            # Draw bounding boxes and keypoints
            result_frame = self.draw_detections(result_frame, filtered_detections)

            # Draw regions of interest
            result_frame = self.draw_roi(result_frame, filtered_detections)

        return result_frame
