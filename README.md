# Hailo Inference Pipeline

![Apache 2.0 License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

![](./images/out_27928552_s.png)![](./images/out_6908_1280x720.gif)

![](./images/out_yolov11.png)![](./images/out_yolov11.gif)

A Python-based inference pipeline for performing deep learning inference using Hailo's hardware acceleration platform.
This project provides a flexible framework for running both synchronous and asynchronous inference on models in Hailo Executable Format (HEF).

## Overview

This project implements an inference pipeline that leverages Hailo's hardware acceleration to process images and videos through deep learning models.
The pipeline supports:

- Synchronous and asynchronous inference modes
- Callback-based result handling
- Image and video input processing
- Multiple model post-processing options
- Batch processing capabilities

## Features

- **Dual Inference Modes**: Choose between synchronous (blocking) or asynchronous (non-blocking) inference
- **Flexible Input Handling**: Process single images or video streams
- **Post-Processing Support**: Built-in support for
  - Hailo Model Zoo classification
  - Hailo Model Zoo object detection (NMS on host cpu only)
  - Mediapipe palm detection model
- **Performance Metrics**: Automatic calculation and display of FPS (Frames Per Second)
- **Resource Management**: Proper cleanup of resources after processing

## Requirements

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- Hailo Platform SDK

## Installation

Follow the PyHailoRT installation procedure of HailoRT documentation.
For running the reference jupyter notebook, you also need to install Hailo DataFlow Compiler to parse/optimize, and compile the tflite model.

## Usage

### Basic Usage

```bash
python inference.py path/to/image.jpg -n path/to/model.hef
```

### Command Line Arguments

| Argument         | Short | Description                                                               | Default                   |
| ---------------- | ----- | ------------------------------------------------------------------------- | ------------------------- |
| `--net`          | `-n`  | Path to the HEF model file                                                | `./hefs/resnet_v1_50.hef` |
| `--postprocess`  | `-p`  | Type of post processing (classification or palm_detection or nms_on_host) | `classification`          |
| `--config`       | `-c`  | Path to model definition JSON file                                        | None                      |
| `--asynchronous` |       | Use asynchronous inference mode                                           | False                     |
| `--callback`     |       | Use callback with asynchronous inference                                  | False                     |
| `--batch-size`   | `-b`  | Number of images in one batch                                             | 1                         |

### Examples

#### Run classification on an image

Download ResNet 50 hef file from the Hailo ModelZoo.
If you don't use Hailo8, please download appropriate hefs from the Hailo ModelZoo and place it under hefs directory.

```bash
wget -O ./hefs/resnet_v1_50.hef https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/resnet_v1_50.hef
```

Then run the script.

```bash
python inference.py -n ./hefs/resnet_v1_50.hef input.jpg
```

#### Process a video with asynchronous inference:

```bash
python inference.py video.mp4 --asynchronous -n ./hefs/resnet_v1_50.hef
```

#### Run classification on an image with a custom label file:

```bash
python inference.py image.jpg -p classification -c my_labels.json -n ./hefs/your_custom_model.hef
```

#### Run object detection an image

Download Yolo hef file from the Hailo ModelZoo. hef must be compiled with "NMS on host CPU" configuration.
Here, we download "Yolov8n" model for detection.
If you don't use Hailo8, please download appropriate hefs from the Hailo ModelZoo and place it under hefs directory.

```bash
wget -O ./hefs/yolov8n.hef https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n.hefs
```

Then run the script.

```bash
python inference.py input.jpg -n ./hefs/yolov8n.hef -p nms_on_host -c ./configs/yolov8.json
```

#### Run palm detection on an image:

Please run the Jupyter Notebook `palm_detection_full_DFC.ipynb` to convert MediaPipe palm detection model to hef.

```bash
cd notebook
jupyter notebook
```

Then run the inference script.

```bash
python inference.py hand.jpg -n ./hef/palm_detection_full.hef -p palm_detection -c ./configs/palm_detection_full.json
```

#### Run palm detection on a video with asynchronous inference and using callback function:

```bash
python inference.py vodeo.mp4 -n ./hef/palm_detection_full.hef -p palm_detection -c ./configs/palm_detection_full.json --asynchronous --callback
```

## How It Works

1. **Initialization**: The pipeline loads a Hailo model and configures input/output streams.
2. **Input Processing**: Images or video frames are resized and padded to match the model's input requirements.
3. **Inference**: The processed input is sent to the Hailo hardware for inference.
4. **Post-Processing**: Results are processed based on the model type (classification or palm detection).
5. **Visualization**: Output is displayed with annotations or classifications.

### Synchronous vs Asynchronous Inference

This pipeline supports two inference modes:

#### Synchronous Inference

- **How it works**: The CPU sends data to the device, waits for processing to complete, then processes the results
- **Pros**: Simple implementation, easier to understand and debug
- **Cons**: Less efficient use of resources as the CPU remains idle during device computation
- **Use case**: Good for simple applications or when you need guaranteed sequential processing

#### Asynchronous Inference

- **How it works**: While the device is processing the current frame, the CPU simultaneously post-processes the previous frame's results
- **Pros**: Higher throughput and better resource utilization, resulting in increased FPS
- **Cons**: More complex implementation and potential for race conditions
- **Use case**: Ideal for video processing or real-time applications where maximum throughput is required

By running CPU post-processing and device inference in parallel, asynchronous inference effectively creates a pipeline that can improve performance, especially in video applications.

## Classes and Components

### InferPipeline

The main class that handles the inference pipeline. Key methods:

- `__init__`: Initialize the pipeline with model and configuration parameters
- `inference`: Run inference on input data
- `wait_and_get_output`: Get results after asynchronous inference completes
- `close`: Clean up resources when done

### Helper Functions

- `preprocess_image_from_array`: Resize images to fit model input requirements
- `preprocess_image_from_array_with_pad`: Resize and pad images while preserving aspect ratio
- `format_tensor_info`: Format tensor information for display

## Post-Processing

The pipeline supports two post-processing options:

1. **Classification**: Maps network outputs to class labels
2. **Palm Detection**: Performs palm detection and visualization

### Model Attribution

- The palm detection model used in this project is from Google MediaPipe: [MediaPipe Models](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md)
- The post-processing code for the MediaPipe model is adapted from [blaze_app_python](https://github.com/AlbertaBeef/blaze_app_python/tree/main)

## About Jupyter Notebooks

This project includes a reference Jupyter notebook that demonstrates:

1. How to build a HEF (Hailo Executable Format) file from a MediaPipe TFLite model
2. How to test inference with the generated HEF file on the notebook

The notebook serves as both a tutorial and a tool to generate HEF files that can be used with the inference Python script in this repository.
This makes it easier to convert and optimize MediaPipe models for use with Hailo hardware acceleration.

To use the notebook:

1. Open it in Jupyter Lab or Jupyter Notebook
2. Follow the step-by-step instructions of DFC notebook to convert your TFLite model to HEF format
3. Run the test inference notebook to verify your model works correctly
4. Use the generated HEF file with the main inference script

## Known issues

There is still race condition with asynchronous inference and sometimes error occurs while video processing.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
