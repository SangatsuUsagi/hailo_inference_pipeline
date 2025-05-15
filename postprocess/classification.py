#!/usr/bin/env python3

import json

import cv2


def add_text_to_image(image, strings):
    """
    Adds a list of text strings to an image at specified positions.

    Parameters:
    - image: numpy array representing the image.
    - strings (list of str): A list of text strings to be added to the image.

    Returns:
    - img: An image object with the text overlay applied. If the image cannot
           be loaded, a ValueError is raised.

    Raises:
    - ValueError: If the image cannot be found or loaded.

    The function right-aligns the text block at the top-right corner of the image,
    maintaining a consistent spacing between lines and starting 10 pixels down from
    the top edge. Each line's position is calculated based on its size to ensure
    alignment.
    """

    # Define properties for the text: font, scale, color (in BGR), and thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)  # White color in BGR format
    thickness = 2

    # Calculate the maximum width and height of all text lines
    text_width = 0
    text_height = 0
    for text in strings:
        ((w, h), _) = cv2.getTextSize(text, font, font_scale, thickness)
        text_width = max(text_width, w)  # Update the widest line width
        text_height = max(text_height, h)  # Update the tallest line height

    # Determine starting positions for right-aligned text at top-right corner
    x_start = image.shape[1] - text_width  # Right-aligned position
    y_start = 10 + text_height // 2  # Start 10 pixels down from the top of the image

    # Draw each line of text on the image
    for i, line in enumerate(strings):
        # Calculate vertical position for this line with consistent spacing
        y_position = y_start + i * (text_height + 10)

        # Add text to the image at the calculated position
        cv2.putText(
            image, line, (x_start, y_position), font, font_scale, color, thickness
        )

    return image


def postprocess(frame, outputs, *args):
    """
    Post-processes model outputs to generate top predictions for each output.

    Args:
        outputs (dict): A dictionary where keys are output names and values are
                        arrays of prediction probabilities.
        args (tuple): Additional arguments expected to contain paths or other parameters.
                      The first argument is expected to be the path to a JSON file
                      containing label mappings.

    Returns:
        None: This function prints the top predictions for each output directly
        to the console.

    Raises:
        FileNotFoundError: If the specified JSON file cannot be found.
        ValueError: If the number of outputs does not match the length of labels
        in the JSON file.
        json.JSONDecodeError: If there is an issue decoding the JSON file.
    """

    # Number of top predictions to be displayed
    TOP_N = 3

    # Generate a dictionary with Top N predicted indices for each output
    top_n_predictions = {k: outputs[k].argsort()[-TOP_N:][::-1] for k in outputs}

    try:
        # Load label list from a JSON file
        with open(args[0], "r") as f:
            labels = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Label file not found at path: {args[0]}. Please provide the correct path."
        )
    except json.JSONDecodeError:
        raise ValueError("Error decoding the label JSON file.")

    # Check if the output length matches the label list length
    for key, probabilities in outputs.items():
        if len(probabilities) != len(labels):
            raise ValueError(
                f"Output length for '{key}' ({len(probabilities)}) does not match with "
                f"label list length ({len(labels)})."
            )

    # Display top N predictions for each output.
    display_string = []
    for key, indices in top_n_predictions.items():
        display_string.append(f"Output: {key}")
        for i, index in enumerate(indices):
            label = labels.get(str(index))
            if label is None:
                raise ValueError(f"Label not found for index '{index}' in JSON file.")
            # Print each of the top N predictions and their corresponding label
            # from the JSON file
            display_string.append(f"#{i + 1}: {label} ({index})")

    add_text_to_image(frame, display_string)

    return frame
