# CLAUDE.md

Guide for AI assistants working on the Hailo Inference Pipeline codebase.

## Project Overview

A Python framework for deploying deep learning models on **Hailo AI hardware accelerators** (Hailo-8, Hailo-8L). Supports image classification (ResNet v1 50), object detection (YOLOv8n with NMS), and palm detection (MediaPipe-adapted). Provides both synchronous and asynchronous inference modes with multi-threaded video processing.

## Repository Structure

```
inference.py           # Main entry point (~1030 lines): CLI, pipeline logic, inference modes
inference_utils.py     # Utility classes: DisplayThread, FrameReaderThread, PerformanceProfiler
postprocess/           # Model-specific post-processing modules
  classification.py    # ImagePostprocessorClassification (ImageNet/ResNet)
  nms_on_host.py       # ImagePostprocessorNmsOnHost (YOLOv8 object detection)
  palm_detection.py    # ImagePostprocessorPalmDetection (hand/palm detection)
configs/               # JSON model configuration files
  yolov8.json          # YOLOv8 detection config (640x640, 80 COCO classes)
  palm_detection_full.json  # Palm detection config (192x192, anchor params)
  class_names_coco.json     # COCO class labels
  class_names_imagenet.json # ImageNet class labels
hefs/                  # HEF model binaries (not tracked in git, *.hef in .gitignore)
notebook/              # Jupyter notebook for model conversion (palm_detection_full_DFC.ipynb)
images/                # Sample images for README documentation
```

## Dependencies

- **Python 3.8+**
- `hailo-platform>=4.0.0` (Hailo SDK, installed separately)
- `opencv-python` (3.x recommended)
- `matplotlib` (performance visualization)
- `numpy`

No `requirements.txt`, `setup.py`, or `pyproject.toml` exists — this is a reference implementation, not a distributable package. Install with:
```bash
pip install opencv-python==3.4.18.65 matplotlib
```

## Running the Application

```bash
python inference.py <image_or_video_path> [OPTIONS]

# Key options:
#   -n, --net FILE           Path to HEF model (default: ./hefs/resnet_v1_50.hef)
#   -p, --postprocess TYPE   classification | nms_on_host | palm_detection
#   -c, --config FILE        Custom JSON config (auto-detected if omitted)
#   -s, --synchronous        Use blocking inference (default: async)
#   --callback               Use callback mode with async
#   -b, --batch-size N       Batch size (default: 1)
#   --profile                Enable performance profiling with charts
```

Requires a physical Hailo device connected via PCIe. Device permissions: `sudo chmod 666 /dev/hailo0`.

## Testing

No formal test suite exists (no pytest, unittest, or CI/CD). Validation is manual — run the pipeline against sample inputs and verify output visually.

## Code Conventions

### Style
- **PEP 8** compliance: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes
- **Type hints on all functions and methods** using `typing` (Optional, Union, Dict, List, Tuple, etc.)
- **Concise docstrings** in Google-style format with Args/Returns/Raises sections — avoid redundant descriptions
- **Comments only for complex logic** — do not add self-evident comments
- **Import grouping**: stdlib → third-party → local

### Architecture Patterns
- **Exception hierarchy**: `InferenceError` base class with `InferenceSubmitError`, `InferenceTimeoutError`, `InferenceWaitError`, `InferencePipelineError`
- **Context manager pattern**: `InferPipeline` uses `__enter__`/`__exit__` for safe resource cleanup
- **Class-based postprocessors**: Each model type has its own postprocessor class with a unified interface (`postprocess(frame, outputs) -> frame`)
- **Thread-based I/O**: `DisplayThread` and `FrameReaderThread` handle video display/capture in separate threads with queue-based communication
- **Configuration via JSON**: Model parameters loaded from `configs/` directory, not hardcoded

### Section Organization in Source Files
- Section dividers use comment blocks with `=` characters (e.g., `# ============`)
- Code ordered: exceptions → classes → helper functions → main

## Key Constants

- `TIMEOUT_MS = 10000` — default inference timeout in milliseconds (in `inference.py`)

## Known Limitations

- Single input tensor models only
- Some Hailo devices don't support synchronous mode (e.g., Hailo-10)
- `--synchronous` and `--callback` flags are mutually exclusive
- Exception handling has not been fully validated
- HEF model files must be obtained/compiled separately (not in repo)

## License

Apache License 2.0
