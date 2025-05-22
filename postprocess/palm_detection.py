#!/usr/bin/env python3

import json
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

"""
Based on code from :
https://github.com/AlbertaBeef/blaze_app_python/blob/main/blaze_common/blazebase.py
"""


class ImagePostprocessorPalmDetection:
    """
    A class to handle the postprocessing of palm detection outputs.

    This class handles anchor generation, detection filtering, coordinate normalization,
    and visualization of palm detection results on input images.

    Attributes:
        scale (float): Scaling factor used during preprocessing.
        pad (tuple): Padding values (x, y) used during preprocessing.
        model_configs (dict): Configuration parameters for the palm detection model.
        anchors (np.ndarray): Generated anchor boxes used for decoding model outputs.
    """

    def __init__(
        self, params: Tuple[Tuple[float, float], Tuple[float, float]], configs: str
    ):
        """
        Initialize the palm detection postprocessor with scaling and padding information.

        Parameters:
            params (tuple): A tuple containing ((x_scale, y_scale), (x_pad, y_pad))
            configs (str): Path to the JSON configuration file containing model information.
        """

        def calculate_scale(
            min_scale: float, max_scale: float, stride_index: int, num_strides: int
        ) -> float:
            """
            Calculate the scale for a specific stride index.

            Parameters:
                min_scale (float): Minimum scale value.
                max_scale (float): Maximum scale value.
                stride_index (int): Current stride index.
                num_strides (int): Total number of strides.

            Returns:
                float: Calculated scale value.
            """
            if num_strides == 1:
                return (max_scale + min_scale) * 0.5
            else:
                return min_scale + (max_scale - min_scale) * stride_index / (
                    num_strides - 1.0
                )

        def generate_anchors(options: Dict[str, Any]) -> np.ndarray:
            """
            Generate anchor boxes based on specified options.

            Parameters:
                options (dict): Dictionary containing anchor generation parameters.
                    Expected keys: strides, num_layers, min_scale, max_scale,
                    reduce_boxes_in_lowest_layer, aspect_ratios, interpolated_scale_aspect_ratio,
                    input_size_height, input_size_width, anchor_offset_x, anchor_offset_y,
                    fixed_anchor_size

            Returns:
                np.ndarray: Generated anchor boxes.
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

                # For same strides, we merge the anchors in the same order.
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

                    if (
                        last_same_stride_layer == 0
                        and options["reduce_boxes_in_lowest_layer"]
                    ):
                        # For first layer, it can be specified to use predefined anchors.
                        aspect_ratios.append(1.0)
                        aspect_ratios.append(2.0)
                        aspect_ratios.append(0.5)
                        scales.append(0.1)
                        scales.append(scale)
                        scales.append(scale)
                    else:
                        for aspect_ratio in options["aspect_ratios"]:
                            aspect_ratios.append(aspect_ratio)
                            scales.append(scale)

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

                for i in range(len(aspect_ratios)):
                    ratio_sqrts = np.sqrt(aspect_ratios[i])
                    anchor_height.append(scales[i] / ratio_sqrts)
                    anchor_width.append(scales[i] * ratio_sqrts)

                stride = options["strides"][layer_id]
                feature_map_height = int(np.ceil(options["input_size_height"] / stride))
                feature_map_width = int(np.ceil(options["input_size_width"] / stride))

                for y in range(feature_map_height):
                    for x in range(feature_map_width):
                        for anchor_id in range(len(anchor_height)):
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

            anchors = np.asarray(anchors)

            return anchors

        # Extract scale and padding from params
        self.scale = params[0][0]  # x_scale
        self.pad = params[1]  # (x_pad, y_pad)

        # Read model configuration and anchor from json file
        try:
            with open(configs, "r", encoding="utf-8") as f:
                model_info = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found at path: {json}. Please provide the correct path."
            )
        except json.JSONDecodeError:
            raise ValueError("Error decoding the label JSON file.")

        # Generate anchor boxes based on configuration options
        self.anchors = generate_anchors(model_info[0])

        # Palm detection configration is listed with anchor and model dictionaties
        self.model_configs = model_info[1]

    def _weighted_non_max_suppression(self, detections: np.ndarray) -> List[np.ndarray]:
        """
        Alternative NMS method as mentioned in the BlazeFace paper.

        This method estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions, rather than using traditional suppression.

        Parameters:
            detections (np.ndarray): Detection tensor of shape (count, 17+).
                The first 4 values in each detection are the bounding box coordinates.
                The value at index num_coords is the score.

        Returns:
            list: A list of filtered detections after blending overlapping predictions.

        Reference:
            mediapipe/calculators/util/non_max_suppression_calculator.cc
            mediapipe/calculators/util/non_max_suppression_calculator.proto
        """

        def intersect(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
            """
            Compute the intersection area between box_a and box_b.

            Parameters:
                box_a (np.ndarray): First set of boxes, shape [A,4].
                box_b (np.ndarray): Second set of boxes, shape [B,4].

            Returns:
                np.ndarray: Intersection areas, shape [A,B].
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

        def jaccard(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
            """
            Compute the Jaccard overlap (IoU) of two sets of boxes.

            Parameters:
                box_a (np.ndarray): First set of boxes, shape [A,4].
                box_b (np.ndarray): Second set of boxes, shape [B,4].

            Returns:
                np.ndarray: Jaccard overlap values, shape [A,B].
            """
            inter = intersect(box_a, box_b)
            area_a = np.repeat(
                np.expand_dims(
                    (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]), axis=1
                ),
                inter.shape[1],
                axis=1,
            )  # [A,B]
            area_b = np.repeat(
                np.expand_dims(
                    (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]), axis=0
                ),
                inter.shape[0],
                axis=0,
            )  # [A,B]
            union = area_a + area_b - inter
            return inter / union  # [A,B]

        def overlap_similarity(box: np.ndarray, other_boxes: np.ndarray) -> np.ndarray:
            """
            Compute the IoU between a bounding box and set of other boxes.

            Parameters:
                box (np.ndarray): Single box, shape [4].
                other_boxes (np.ndarray): Set of boxes to compare against, shape [N,4].

            Returns:
                np.ndarray: IoU values between the box and all other_boxes.
            """
            return jaccard(np.expand_dims(box, axis=0), other_boxes).squeeze(0)

        if len(detections) == 0:
            return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        # argsort() returns ascending order, therefore read the array from end
        remaining = np.argsort(detections[:, self.model_configs["num_coords"]])[::-1]

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.model_configs["min_suppression_threshold"]
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
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

    def denormalize_detections(self, detections: np.ndarray) -> np.ndarray:
        """
        Maps detection coordinates from [0,1] to original image coordinates.

        The face and palm detector networks take 256x256 and 128x128 images
        as input. As such the input image is padded and resized to fit the
        size while maintaining the aspect ratio. This function maps the
        normalized coordinates back to the original image coordinates.

        Parameters:
            detections (np.ndarray): Normalized detection coordinates.

        Returns:
            np.ndarray: Denormalized detection coordinates in original image space.
        """
        detections[:, 0] = (
            detections[:, 0] * self.scale * self.model_configs["x_scale"] - self.pad[0]
        )
        detections[:, 1] = (
            detections[:, 1] * self.scale * self.model_configs["x_scale"] - self.pad[1]
        )
        detections[:, 2] = (
            detections[:, 2] * self.scale * self.model_configs["x_scale"] - self.pad[0]
        )
        detections[:, 3] = (
            detections[:, 3] * self.scale * self.model_configs["x_scale"] - self.pad[1]
        )

        detections[:, 4::2] = (
            detections[:, 4::2] * self.scale * self.model_configs["x_scale"]
            - self.pad[1]
        )
        detections[:, 5::2] = (
            detections[:, 5::2] * self.scale * self.model_configs["x_scale"]
            - self.pad[0]
        )

        return detections

    def postprocess_palm_detection(
        self, outputs: Dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        """
        Post-processes raw model outputs to generate filtered detections for palm positions.

        Parameters:
            outputs (dict): A dictionary containing the output tensors from the neural network.
                            Expected keys are 'palm_detection_full/conv29', 'palm_detection_full/conv34',
                            'palm_detection_full/conv30', and 'palm_detection_full/conv35'.

        Returns:
            list: A list containing filtered detections after applying weighted non-maximum suppression.
                If no valid detections are found, returns an empty list.

        Raises:
            ValueError: If required output tensors are missing or if there's an error reshaping tensors.
        """

        def _decode_boxes(raw_boxes: np.ndarray) -> np.ndarray:
            """
            Converts the predictions into actual coordinates using the anchor boxes.

            Parameters:
                raw_boxes (np.ndarray): Raw box predictions from the model.

            Returns:
                np.ndarray: Decoded box coordinates.
            """
            boxes = np.zeros(raw_boxes.shape)

            # Convert center coordinates using anchor boxes
            x_center = (
                raw_boxes[..., 0] / self.model_configs["x_scale"] * self.anchors[:, 2]
                + self.anchors[:, 0]
            )
            y_center = (
                raw_boxes[..., 1] / self.model_configs["y_scale"] * self.anchors[:, 3]
                + self.anchors[:, 1]
            )

            # Convert width and height using anchor boxes
            w = raw_boxes[..., 2] / self.model_configs["w_scale"] * self.anchors[:, 2]
            h = raw_boxes[..., 3] / self.model_configs["h_scale"] * self.anchors[:, 3]

            # Calculate box coordinates from center and dimensions
            boxes[..., 0] = y_center - h / 2.0  # ymin
            boxes[..., 1] = x_center - w / 2.0  # xmin
            boxes[..., 2] = y_center + h / 2.0  # ymax
            boxes[..., 3] = x_center + w / 2.0  # xmax

            # Convert keypoint coordinates using anchor boxes
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
            raw_box_tensor: np.ndarray, raw_score_tensor: np.ndarray
        ) -> List[np.ndarray]:
            """
            Converts raw network outputs into proper detections.

            Parameters:
                raw_box_tensor (np.ndarray): Raw bounding box tensor from the network.
                raw_score_tensor (np.ndarray): Raw score tensor from the network.

            Returns:
                list: A list of detection arrays with bounding boxes and scores.

            Reference:
                mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
                mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
            """
            # Decode boxes from raw outputs
            detection_boxes = _decode_boxes(raw_box_tensor)

            # Clip and apply sigmoid to scores
            thresh = self.model_configs["score_clipping_thresh"]
            clipped_score_tensor = np.clip(raw_score_tensor, -thresh, thresh)
            detection_scores = 1 / (1 + np.exp(-clipped_score_tensor))
            detection_scores = np.squeeze(detection_scores, axis=-1)

            # Filter out boxes with low confidence
            mask = detection_scores >= self.model_configs["min_score_thresh"]

            # Process each image in the batch
            output_detections = []
            for i in range(raw_box_tensor.shape[0]):
                boxes = detection_boxes[i, mask[i]]
                scores = detection_scores[i, mask[i]]
                scores = np.expand_dims(scores, axis=-1)
                boxes_scores = np.concatenate((boxes, scores), axis=-1)
                output_detections.append(boxes_scores)

            return output_detections

        # Validate inputs
        required_keys = [
            "palm_detection_full/conv29",
            "palm_detection_full/conv34",
            "palm_detection_full/conv30",
            "palm_detection_full/conv35",
        ]
        for key in required_keys:
            if key not in outputs:
                raise ValueError(f"Missing expected output tensor: {key}")

        # Reshape specific convolutional layers to prepare for concatenation
        try:
            # Score tensors
            conv2D_1 = np.reshape(outputs["palm_detection_full/conv29"], (1, 864, 1))
            conv2D_2 = np.reshape(outputs["palm_detection_full/conv34"], (1, 1152, 1))
            # Box coordinate tensors
            conv2D_3 = np.reshape(outputs["palm_detection_full/conv30"], (1, 864, 18))
            conv2D_4 = np.reshape(outputs["palm_detection_full/conv35"], (1, 1152, 18))
        except ValueError as e:
            raise ValueError(f"Error reshaping tensors: {e}")

        # Concatenate reshaped tensors along the second axis for scores
        out1 = np.concatenate((conv2D_2, conv2D_1), axis=1)

        # Concatenate along the second axis for box coordinates
        out2 = np.concatenate((conv2D_4, conv2D_3), axis=1)

        # Validate the shapes of concatenated tensors against expected configurations
        assert out1.shape[0] == 1  # batch size must be 1
        assert out1.shape[1] == self.model_configs["num_anchors"]  # number of anchors
        assert out1.shape[2] == 1  # single score per anchor

        assert out2.shape[0] == 1  # batch size must be 1
        assert out2.shape[1] == self.model_configs["num_anchors"]  # number of anchors
        assert (
            out2.shape[2] == self.model_configs["num_coords"]
        )  # coordinates per anchor

        # Convert tensors to detection format using model-specific logic
        detections = _tensors_to_detections(out2, out1)

        # Apply weighted non-maximum suppression to remove overlapping detections
        filtered_detections = []
        for i in range(len(detections)):
            wnms_detections = self._weighted_non_max_suppression(detections[i])
            if len(wnms_detections) > 0:
                filtered_detections.append(wnms_detections)

        # Normalize final detection list
        if len(filtered_detections) > 0:
            normalized_detections = np.array(filtered_detections)[0]
        else:
            normalized_detections = []

        return normalized_detections

    def draw_detections(
        self, image: np.ndarray, filtered_detections: np.ndarray
    ) -> np.ndarray:
        """
        Draws the filtered detections on an image by marking detected palms and keypoints.

        Parameters:
            image (np.ndarray): The input image on which to draw the detections.
            filtered_detections (np.ndarray): A list of detection coordinates after non-maximum suppression.

        Returns:
            np.ndarray: The input image with drawn detections.

        Raises:
            TypeError: If the input image is not a numpy ndarray.
        """

        # Ensure the input image is a NumPy array
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a numpy ndarray")

        if len(filtered_detections) > 0:
            # Denormalize detection coordinates to fit the original image scale and padding
            detections = self.denormalize_detections(filtered_detections)

            for i in range(detections.shape[0]):
                ymin, xmin, ymax, xmax = detections[i, :4]

                # Draw bounding box around detected palm using OpenCV
                top_left = (int(xmin), int(ymin))
                bottom_right = (int(xmax), int(ymax))
                cv2.rectangle(
                    image, top_left, bottom_right, color=(0, 0, 255), thickness=4
                )

                # Draw keypoints as circles
                n_keypoints = detections.shape[1] // 2 - 2
                for k in range(n_keypoints):
                    kp_x = int(detections[i, 4 + k * 2])
                    kp_y = int(detections[i, 4 + k * 2 + 1])
                    radius = 10

                    # Draw keypoints as circles using OpenCV
                    cv2.circle(
                        image,
                        (kp_x, kp_y),
                        radius=radius,
                        color=(0, 0, 255),
                        thickness=2,
                    )

        return image

    def detection2roi(
        self, detection: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert detections from detector to an oriented bounding box.

        The center and size of the box is calculated from the center
        of the detected box. Rotation is calculated from the vector
        between kp1 and kp2 relative to theta0. The box is scaled
        and shifted by dscale and dy.

        Parameters:
            detection (np.ndarray): Detection array containing bounding box and keypoints.

        Returns:
            tuple: (xc, yc, scale, theta) containing center coordinates, scale, and rotation.

        Reference:
            mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
        """
        # Compute box center and scale
        # Use mediapipe/calculators/util/detections_to_rects_calculator.cc
        xc = (detection[:, 1] + detection[:, 3]) / 2
        yc = (detection[:, 0] + detection[:, 2]) / 2
        scale = detection[:, 3] - detection[:, 1]  # assumes square boxes

        # Apply shift and scale adjustments from model config
        yc += self.model_configs["dy"] * scale
        scale *= self.model_configs["dscale"]

        # Compute box rotation from keypoints
        x0 = detection[:, 4 + 2 * self.model_configs["kp1"]]
        y0 = detection[:, 4 + 2 * self.model_configs["kp1"] + 1]
        x1 = detection[:, 4 + 2 * self.model_configs["kp2"]]
        y1 = detection[:, 4 + 2 * self.model_configs["kp2"] + 1]
        theta = np.arctan2(y0 - y1, x0 - x1) - self.model_configs["theta0"]

        return xc, yc, scale, theta

    def extract_roi(
        self, xc: np.ndarray, yc: np.ndarray, theta: np.ndarray, scale: np.ndarray
    ) -> np.ndarray:
        """
        Extracts regions of interest (ROIs) by applying transformations to points.

        Parameters:
            xc (np.ndarray): Array of x-coordinates for the centers of ROIs.
            yc (np.ndarray): Array of y-coordinates for the centers of ROIs.
            theta (np.ndarray): Array of angles in radians for rotation of each ROI.
            scale (np.ndarray): Array of scaling factors to resize each ROI.

        Returns:
            np.ndarray: Transformed points representing the corners of each ROI.
                        Shape is (n, 2, 4) where n is the number of ROIs.
        """

        # Reshape scale array for broadcasting during scaling operations
        scaleN = scale.reshape(-1, 1, 1).astype(np.float32)

        # Define the base square points (corners of a unit square centered at origin)
        points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=np.float32)

        # Apply scaling to each point and normalize
        points = points * scaleN / 2
        points = points.astype(np.float32)

        # Initialize rotation matrices for each ROI
        R = np.zeros((theta.shape[0], 2, 2), dtype=np.float32)

        # Populate rotation matrices based on theta angles
        for i in range(theta.shape[0]):
            R[i, :, :] = [
                [np.cos(theta[i]), -np.sin(theta[i])],
                [np.sin(theta[i]), np.cos(theta[i])],
            ]

        # Stack center coordinates for translation
        center = np.column_stack((xc, yc))
        center = np.expand_dims(center, axis=-1)

        # Apply rotation and translation to points
        points = np.matmul(R, points) + center
        points = points.astype(np.float32)

        return points

    def draw_roi(
        self, image: np.ndarray, filtered_detections: np.ndarray
    ) -> np.ndarray:
        """
        Draws regions of interest (ROIs) on a given image.

        Parameters:
            image (np.ndarray): The input image on which to draw ROIs.
            filtered_detections (np.ndarray): Filtered detection coordinates.

        Returns:
            np.ndarray: The image with ROIs drawn on it.

        Raises:
            TypeError: If the input image is not a numpy ndarray.
        """

        # Ensure the input image is a NumPy array
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a numpy ndarray")

        # Convert detections to ROI parameters
        xc, yc, scale, theta = self.detection2roi(filtered_detections)

        # Extract ROI corner points
        roi_box = self.extract_roi(xc, yc, theta, scale)

        # Draw ROI quadrilaterals on the image
        for i in range(roi_box.shape[0]):
            # Extract the (x, y) coordinates for the current ROI
            (x1, x2, x3, x4), (y1, y2, y3, y4) = roi_box[i]

            # Convert floating point coordinates to integers for drawing
            pt1_start = (int(x1), int(y1))  # First corner of the quadrilateral
            pt2_mid_top = (int(x2), int(y2))  # Second corner at mid-top
            pt2_mid_bottom = (int(x3), int(y3))  # Third corner at mid-bottom
            pt3_end = (int(x4), int(y4))  # Fourth and final corner

            # Draw lines using OpenCV to form the quadrilateral
            cv2.line(image, pt1_start, pt2_mid_top, color=(255, 0, 0), thickness=3)
            cv2.line(image, pt1_start, pt2_mid_bottom, color=(255, 0, 0), thickness=3)
            cv2.line(image, pt2_mid_top, pt3_end, color=(255, 0, 0), thickness=3)
            cv2.line(image, pt2_mid_bottom, pt3_end, color=(255, 0, 0), thickness=3)

        return image

    def postprocess(
        self, frame: np.ndarray, outputs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Main postprocessing function that processes model outputs and visualizes results.

        Parameters:
            frame (np.ndarray): The input image frame.
            outputs (dict): Model output tensors containing detection results.

        Returns:
            np.ndarray: The processed frame with palm detections and ROIs visualized.
        """
        # Perform palm detection postprocessing
        filtered_detections = self.postprocess_palm_detection(outputs)

        if len(filtered_detections) != 0:
            # Draw palm detections on the frame
            frame = self.draw_detections(frame, filtered_detections)

            # Draw regions of interest on the frame
            frame = self.draw_roi(frame, filtered_detections)

        return frame
