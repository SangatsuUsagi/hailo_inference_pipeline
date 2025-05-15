#!/usr/bin/env python3

import cv2
import numpy as np

"""
Based on code from :
https://github.com/AlbertaBeef/blaze_app_python/blob/main/blaze_common/blazebase.py
"""


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (max_scale + min_scale) * 0.5
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


def generate_anchors(options):
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
            options["strides"][last_same_stride_layer] == options["strides"][layer_id]
        ):
            scale = calculate_scale(
                options["min_scale"],
                options["max_scale"],
                last_same_stride_layer,
                strides_size,
            )

            if last_same_stride_layer == 0 and options["reduce_boxes_in_lowest_layer"]:
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
                    aspect_ratios.append(options["interpolated_scale_aspect_ratio"])

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
                    x_center = (x + options["anchor_offset_x"]) / feature_map_width
                    y_center = (y + options["anchor_offset_y"]) / feature_map_height

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


def _decode_boxes(raw_boxes, anchors, configs):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    boxes = np.zeros(raw_boxes.shape)

    x_center = raw_boxes[..., 0] / configs["x_scale"] * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / configs["y_scale"] * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / configs["w_scale"] * anchors[:, 2]
    h = raw_boxes[..., 3] / configs["h_scale"] * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.0  # ymin
    boxes[..., 1] = x_center - w / 2.0  # xmin
    boxes[..., 2] = y_center + h / 2.0  # ymax
    boxes[..., 3] = x_center + w / 2.0  # xmax

    for k in range(configs["num_keypoints"]):
        offset = 4 + k * 2
        keypoint_x = (
            raw_boxes[..., offset] / configs["x_scale"] * anchors[:, 2] + anchors[:, 0]
        )
        keypoint_y = (
            raw_boxes[..., offset + 1] / configs["y_scale"] * anchors[:, 3]
            + anchors[:, 1]
        )
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


def _tensors_to_detections(raw_box_tensor, raw_score_tensor, anchors, configs):
    """The output of the neural network is an array of shape (b, 896, 12)
    containing the bounding box regressor predictions, as well as an array
    of shape (b, 896, 1) with the classification confidences.

    This function converts these two "raw" arrays into proper detections.
    Returns a list of (num_detections, 13) arrays, one for each image in
    the batch.

    This is based on the source code from:
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
    """
    detection_boxes = _decode_boxes(raw_box_tensor, anchors, configs)

    thresh = configs["score_clipping_thresh"]
    clipped_score_tensor = np.clip(raw_score_tensor, -thresh, thresh)

    detection_scores = 1 / (1 + np.exp(-clipped_score_tensor))
    detection_scores = np.squeeze(detection_scores, axis=-1)

    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = detection_scores >= configs["min_score_thresh"]

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i, mask[i]]

        scores = detection_scores[i, mask[i]]
        scores = np.expand_dims(scores, axis=-1)

        boxes_scores = np.concatenate((boxes, scores), axis=-1)
        output_detections.append(boxes_scores)

    return output_detections


def intersect(box_a, box_b):
    """We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
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


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
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


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(np.expand_dims(box, axis=0), other_boxes).squeeze(0)


def _weighted_non_max_suppression(detections, configs):
    """The alternative NMS method as mentioned in the BlazeFace paper:

    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.

    The input detections should be a Tensor of shape (count, 17).

    Returns a list of PyTorch tensors, one for each detected face.

    This is based on the source code from:
    mediapipe/calculators/util/non_max_suppression_calculator.cc
    mediapipe/calculators/util/non_max_suppression_calculator.proto
    """
    if len(detections) == 0:
        return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    # argsort() returns ascending order, therefore read the array from end
    remaining = np.argsort(detections[:, configs["num_coords"]])[::-1]

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
        mask = ious > configs["min_suppression_threshold"]
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.copy()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, : configs["num_coords"]]
            scores = detections[
                overlapping, configs["num_coords"] : configs["num_coords"] + 1
            ]
            total_score = scores.sum()
            weighted = np.sum(coordinates * scores, axis=0) / total_score
            weighted_detection[: configs["num_coords"]] = weighted
            weighted_detection[configs["num_coords"]] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections


def denormalize_detections(detections, scale, pad, configs):
    """maps detection coordinates from [0,1] to image coordinates

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.

    Inputs:
        detections: nxm tensor. n is the number of detections.
            m is 4+2*k where the first 4 valuse are the bounding
            box coordinates and k is the number of additional
            keypoints output by the detector.
        scale: scalar that was used to resize the image
        pad: padding in the x and y dimensions

    """
    detections[:, 0] = detections[:, 0] * scale * configs["x_scale"] - pad[0]
    detections[:, 1] = detections[:, 1] * scale * configs["x_scale"] - pad[1]
    detections[:, 2] = detections[:, 2] * scale * configs["x_scale"] - pad[0]
    detections[:, 3] = detections[:, 3] * scale * configs["x_scale"] - pad[1]

    detections[:, 4::2] = detections[:, 4::2] * scale * configs["x_scale"] - pad[1]
    detections[:, 5::2] = detections[:, 5::2] * scale * configs["x_scale"] - pad[0]
    return detections


def postprocess_palm_detection(outputs, anchor_params, configs):
    """
    Post-processes raw model outputs to generate filtered detections for palm positions in images.

    Parameters:
        outputs (dict): A dictionary containing the output tensors from a neural network.
                        Expected keys are 'palm_detection_full/conv29', 'palm_detection_full/conv34',
                        'palm_detection_full/conv30', and 'palm_detection_full/conv35'.
        configs (dict): Configuration settings, which include 'num_anchors' for the number of anchors,
                        'num_coords' for the number of coordinates per anchor, and parameters needed
                        for non-maximum suppression.

    Returns:
        list: A list containing filtered detections after applying weighted non-maximum suppression.
              If no valid detections are found, returns an empty list.
    """

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
        conv2D_1 = np.reshape(outputs["palm_detection_full/conv29"], (1, 864, 1))
        conv2D_2 = np.reshape(outputs["palm_detection_full/conv34"], (1, 1152, 1))
        conv2D_3 = np.reshape(outputs["palm_detection_full/conv30"], (1, 864, 18))
        conv2D_4 = np.reshape(outputs["palm_detection_full/conv35"], (1, 1152, 18))
    except ValueError as e:
        raise ValueError("Error reshaping tensors: " + str(e))

    # Concatenate reshaped tensors along the second axis and select the first element in batch
    out1 = np.concatenate((conv2D_2, conv2D_1), axis=1)

    # Concatenate along the second axis for another set of outputs
    out2 = np.concatenate((conv2D_4, conv2D_3), axis=1)

    # Generate anchor boxes based on configuration options
    anchors = generate_anchors(anchor_params)

    # Validate the shapes of concatenated tensors against expected configurations
    assert out1.shape[0] == 1  # batch size must be 1
    assert out1.shape[1] == configs["num_anchors"]  # number of anchors
    assert out1.shape[2] == 1  # single score per anchor

    assert out2.shape[0] == 1  # batch size must be 1
    assert out2.shape[1] == configs["num_anchors"]  # number of anchors
    assert out2.shape[2] == configs["num_coords"]  # coordinates per anchor

    # Convert tensors to detection format using model-specific logic
    detections = _tensors_to_detections(out2, out1, anchors, configs)

    # Apply weighted non-maximum suppression to remove overlapping detections
    filtered_detections = []
    for i in range(len(detections)):
        wnms_detections = _weighted_non_max_suppression(detections[i], configs)
        if len(wnms_detections) > 0:
            filtered_detections.append(wnms_detections)

    # Normalize final detection list
    if len(filtered_detections) > 0:
        normalized_detections = np.array(filtered_detections)[0]
    else:
        normalized_detections = []

    return normalized_detections


def draw_detections(image, filtered_detections, scale, pad, configs):
    """
    Draws the filtered detections on an image by marking detected palms and keypoints.

    Parameters:
        filtered_detections (list): A list of detection coordinates after non-maximum suppression.
        image (numpy.ndarray): The image on which to draw the detections.
        scale (float): Scaling factor applied to detection coordinates.
        pad (tuple or list): Padding values for scaling and offset adjustments.
        configs (dict): Configuration settings, including 'num_coords' used for keypoint calculations.

    Returns:
        numpy.ndarray: The input image with drawn detections.
    """

    # Ensure the input image is a NumPy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be a numpy ndarray")

    if len(filtered_detections) > 0:
        # Denormalize detection coordinates to fit the original image scale and padding
        detections = denormalize_detections(filtered_detections, scale, pad, configs)

        for i in range(detections.shape[0]):
            ymin, xmin, ymax, xmax = detections[i, :4]

            # Draw bounding box around detected palm using OpenCV
            top_left = (int(xmin), int(ymin))
            bottom_right = (int(xmax), int(ymax))
            cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 255), thickness=4)

            n_keypoints = detections.shape[1] // 2 - 2
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k * 2])
                kp_y = int(detections[i, 4 + k * 2 + 1])
                radius = 10

                # Draw keypoints as circles using OpenCV
                cv2.circle(
                    image, (kp_x, kp_y), radius=radius, color=(0, 0, 255), thickness=2
                )

    return image


def detection2roi(detection, configs):
    """Convert detections from detector to an oriented bounding box.

    Adapted from:
    # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

    The center and size of the box is calculated from the center
    of the detected box. Rotation is calcualted from the vector
    between kp1 and kp2 relative to theta0. The box is scaled
    and shifted by dscale and dy.

    """
    # compute box center and scale
    # use mediapipe/calculators/util/detections_to_rects_calculator.cc
    xc = (detection[:, 1] + detection[:, 3]) / 2
    yc = (detection[:, 0] + detection[:, 2]) / 2
    scale = detection[:, 3] - detection[:, 1]  # assumes square boxes

    yc += configs["dy"] * scale
    scale *= configs["dscale"]

    # compute box rotation
    x0 = detection[:, 4 + 2 * configs["kp1"]]
    y0 = detection[:, 4 + 2 * configs["kp1"] + 1]
    x1 = detection[:, 4 + 2 * configs["kp2"]]
    y1 = detection[:, 4 + 2 * configs["kp2"] + 1]
    theta = np.arctan2(y0 - y1, x0 - x1) - configs["theta0"]

    return xc, yc, scale, theta


def extract_roi(xc, yc, theta, scale):
    """
    Extracts regions of interest (ROIs) by applying scaling, rotation,
    and translation transformations to a set of predefined points.

    Parameters:
    xc (np.ndarray): Array of x-coordinates for the centers of ROIs.
    yc (np.ndarray): Array of y-coordinates for the centers of ROIs.
    theta (np.ndarray): Array of angles in radians for rotation of each ROI.
    scale (np.ndarray): Array of scaling factors, size [N], to resize each ROI.

    Returns:
    np.ndarray: Transformed points representing the corners of each ROI after
                applying the specified transformations.

    The function assumes that the input arrays `xc`, `yc`, and `theta` have
    compatible dimensions for batch processing. Specifically, they should all be
    of size [N], where N is the number of ROIs to process. The scale array should
    also be of size [N].
    """

    # Reshape scale array to allow broadcasting during scaling operations.
    # The new shape will be [N, 1, 1] for element-wise multiplication with points.
    scaleN = scale.reshape(-1, 1, 1).astype(np.float32)

    # Define the base square points (corners of a unit square centered at origin).
    points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=np.float32)

    # Apply scaling to each point and normalize by dividing by 2.
    # This centers the scaled points around the origin before rotation and translation.
    points = points * scaleN / 2
    points = points.astype(np.float32)  # Ensure points are of type float32.

    # Initialize an array for rotation matrices, one for each ROI.
    R = np.zeros((theta.shape[0], 2, 2), dtype=np.float32)

    # Populate the rotation matrix for each ROI based on its angle theta[i].
    for i in range(theta.shape[0]):
        R[i, :, :] = [
            [np.cos(theta[i]), -np.sin(theta[i])],
            [np.sin(theta[i]), np.cos(theta[i])],
        ]

    # Stack x and y center coordinates into a single array and add an extra dimension
    # for broadcasting in the upcoming matrix multiplication.
    center = np.column_stack((xc, yc))
    center = np.expand_dims(center, axis=-1)

    # Apply rotation to points using matrix multiplication with rotation matrices R.
    # Add translation by adding the centers (xc, yc) to each transformed point set.
    points = np.matmul(R, points) + center
    points = points.astype(np.float32)  # Ensure points are of type float32.

    return points


def draw_roi(image, filtered_detections, model_configs):
    """
    Draws regions of interest (ROIs) on a given image.

    This function takes an input image and a set of ROI coordinates,
    then draws quadrilateral shapes on the image using specified points.

    Parameters:
    -----------
    image : np.ndarray
        The input image represented as a NumPy array. It is expected to be in BGR format.

    roi : np.ndarray
        An array containing regions of interest, where each ROI consists
        of four (x, y) coordinate pairs representing the corners of a quadrilateral.

    Returns:
    --------
    np.ndarray
        The image with the ROIs drawn on it. The output is in BGR format as well.

    Raises:
    -------
    TypeError
        If the input image is not an instance of NumPy ndarray.
    """

    # Ensure the input image is a NumPy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be a numpy ndarray")

    xc, yc, scale, theta = detection2roi(filtered_detections, model_configs)
    roi_box = extract_roi(xc, yc, theta, scale)

    for i in range(roi_box.shape[0]):
        # Extract the (x, y) coordinates for the current ROI
        (x1, x2, x3, x4), (y1, y2, y3, y4) = roi_box[i]

        # Convert floating point coordinates to integers for drawing purposes
        pt1_start = (int(x1), int(y1))  # First corner of the quadrilateral
        pt2_mid_top = (int(x2), int(y2))  # Second corner at mid-top
        pt2_mid_bottom = (int(x3), int(y3))  # Third corner at mid-bottom
        pt3_end = (int(x4), int(y4))  # Fourth and final corner

        # Draw lines using OpenCV to form the quadrilateral on the image
        cv2.line(image, pt1_start, pt2_mid_top, color=(255, 0, 0), thickness=3)
        cv2.line(image, pt1_start, pt2_mid_bottom, color=(255, 0, 0), thickness=3)
        cv2.line(image, pt2_mid_top, pt3_end, color=(255, 0, 0), thickness=3)
        cv2.line(image, pt2_mid_bottom, pt3_end, color=(255, 0, 0), thickness=3)

    return image


def postprocess(frame, outputs, *args):
    # reference : https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/palm_detection/palm_detection_gpu.pbtxt
    anchor_options = {
        "num_layers": 4,
        "min_scale": 0.1484375,
        "max_scale": 0.75,
        "input_size_height": 192,
        "input_size_width": 192,
        "anchor_offset_x": 0.5,
        "anchor_offset_y": 0.5,
        "strides": [8, 16, 16, 16],
        "aspect_ratios": [1.0],
        "reduce_boxes_in_lowest_layer": False,
        "interpolated_scale_aspect_ratio": 1.0,
        "fixed_anchor_size": True,
    }

    model_configs = {
        "num_classes": 1,
        "num_anchors": 2016,
        "num_coords": 18,
        "score_clipping_thresh": 100.0,
        "x_scale": 192.0,
        "y_scale": 192.0,
        "h_scale": 192.0,
        "w_scale": 192.0,
        "min_score_thresh": 0.5,
        "min_suppression_threshold": 0.3,
        "num_keypoints": 7,
        "kp1": 0,
        "kp2": 2,
        "theta0": np.pi / 2,
        "dscale": 2.6,
        "dy": -0.5,
        "resolution": 256,
    }

    scale = args[0]
    pad = args[1]

    filtered_detections = postprocess_palm_detection(
        outputs, anchor_options, model_configs
    )

    if len(filtered_detections) != 0:
        # Draw detections
        frame = draw_detections(frame, filtered_detections, scale, pad, model_configs)

        # Draw ROI
        frame = draw_roi(frame, filtered_detections, model_configs)

    return frame
