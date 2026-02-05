---
license: openrail
base_model: Supertone/supertonic-2
tags:
  - tts
  - text-to-speech
  - qualcomm
  - qnn
  - quantized
  - qcs6490
  - hexagon
pipeline_tag: text-to-speech
---

# Supertonic TTS Quantization for QCS6490

A step-by-step guide to quantize the [Supertonic TTS](https://huggingface.co/Supertone/supertonic) model for Qualcomm QCS6490 using QAIRT/QNN.

## Requirements

- QAIRT/QNN SDK **v2.37**
- Python 3.8+
- Target device: **QCS6490**

## Pipeline Architecture

```text
text_encoder → vector_estimator (10 diffusion steps) → vocoder → audio
```

## Workflow

### 1. Input Preparation

Prepare calibration inputs for model quantization.

`Input_Preparation.ipynb`

### 2. Step-by-Step Quantization

Convert ONNX models to QNN format with quantization for HTP backend.

`Supertonic_TTS_StepbyStep.ipynb`

### 3. Correlation Verification

Verify quantized model outputs against reference using cosine similarity.

`Correlation_Verification.ipynb`

## Project Structure

```text
├── Input_Preparation.ipynb         # Prepare calibration inputs
├── Supertonic_TTS_StepbyStep.ipynb # ONNX → QNN quantization guide
├── Correlation_Verification.ipynb  # Output verification
├── assets/                         # ONNX models (git submodule)
│   └── onnx/
│       ├── text_encoder.onnx
│       ├── duration_predictor.onnx
│       ├── vector_estimator.onnx
│       └── vocoder.onnx
├── QNN_Models/                     # Quantized QNN models (.bin, .cpp)
├── QNN_Model_lib/                  # QNN runtime libraries (aarch64)
├── qnn_calibration/                # Calibration data for verification
├── inputs/                         # Prepared input data
└── board_output/                   # Inference outputs from board
```

## Models

| Model              | Description                                 |
|--------------------|---------------------------------------------|
| text_encoder       | Encodes text tokens with style embedding    |
| duration_predictor | Predicts phoneme durations                  |
| vector_estimator   | Diffusion-based latent generator (10 steps) |
| vocoder            | Converts latent to audio waveform           |

### ONNX Models (Source)

Located in `assets/onnx/` (git submodule from Hugging Face):

- `text_encoder.onnx`
- `duration_predictor.onnx`
- `vector_estimator.onnx`
- `vocoder.onnx`

### QNN Models (Quantized)

Located in `QNN_Models/`:

- `text_encoder_htp.bin` / `.cpp`
- `vector_estimator_htp.bin` / `.cpp`
- `vocoder_htp.bin` / `.cpp`

### Compiled Libraries (Ready for Deployment)

Located in `QNN_Model_lib/aarch64-oe-linux-gcc11.2/`:

- `libtext_encoder_htp.so`
- `libvector_estimator_htp.so`
- `libvocoder_htp.so`
- `libduration_predictor_htp.so`

These `.so` files are compiled from the `.cpp` sources and are ready to be deployed (via SCP) to the board for inference.

> **Note:** The `duration_predictor` model is not quantized as all its layers are static. Duration prediction is not required by the split models.

## Getting Started

1. Clone with submodules:

   ```bash
   git clone --recurse-submodules https://github.com/dev-ansh-r/Supertonic-TTS-QCS6490
   ```

2. Follow the notebooks in order:
   - `Input_Preparation.ipynb`
   - `Supertonic_TTS_StepbyStep.ipynb`
   - `Correlation_Verification.ipynb`

## Note

> Inference script and sample application are not provided. Optimization work is ongoing and will be released soon.

## License

This model inherits the licensing from [Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2):

- **Model:** OpenRAIL-M License
- **Code:** MIT License

Copyright (c) 2026 Supertone Inc. (original model)
