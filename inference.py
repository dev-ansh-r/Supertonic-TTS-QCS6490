#!/usr/bin/env python3
"""
TTS Inference Pipeline for QCS6490
Models: text_encoder -> vector_estimator (10 steps) -> vocoder

Usage:
    python inference.py --prepare    # Prepare inputs from calibration data
    python inference.py --run        # Run on board (requires SSH access)
    python inference.py --convert    # Convert output to WAV
"""

import numpy as np
import os
import argparse
import subprocess

# Quantization parameters from model .cpp files
QUANT_PARAMS = {
    # Text Encoder inputs
    'text_ids': {'dtype': 'int32', 'shape': (1, 128)},
    'style_ttl_te': {'dtype': 'uint8', 'shape': (1, 256, 50), 'onnx_shape': (1, 50, 256),
                     'scale': 0.0054120589047670, 'offset': -172},
    'text_mask_te': {'dtype': 'uint8', 'shape': (1, 128, 1), 'onnx_shape': (1, 1, 128),
                     'scale': 0.0039215688593686, 'offset': 0},

    # Vector Estimator inputs
    'noisy_latent': {'dtype': 'uint8', 'shape': (1, 256, 144), 'onnx_shape': (1, 144, 256),
                     'scale': 0.0322749949991703, 'offset': -127},
    'text_emb': {'dtype': 'uint8', 'shape': (1, 128, 256),
                 'scale': 0.0278750918805599, 'offset': -128},
    'style_ttl_ve': {'dtype': 'uint8', 'shape': (1, 256, 50), 'onnx_shape': (1, 50, 256),
                     'scale': 0.0054120589047670, 'offset': -172},
    'latent_mask': {'dtype': 'uint8', 'shape': (1, 256, 1), 'onnx_shape': (1, 1, 256),
                    'scale': 0.0039215688593686, 'offset': 0},
    'text_mask_ve': {'dtype': 'uint8', 'shape': (1, 128, 1), 'onnx_shape': (1, 1, 128),
                     'scale': 0.0039215688593686, 'offset': 0},
    'current_step': {'dtype': 'uint8', 'shape': (1,),
                     'scale': 0.0000003921568634, 'offset': 0},
    'total_step': {'dtype': 'uint8', 'shape': (1,),
                   'scale': 0.0196078438311815, 'offset': 0},

    # Vocoder inputs
    'latent': {'dtype': 'uint8', 'shape': (1, 256, 144),
               'scale': 0.0272593162953854, 'offset': -129},
}

# Directories
INPUT_DIR = './inputs'
OUTPUT_DIR = './board_output'
CALIB_DIR = './qnn_calibration'

def quantize(data, scale, offset):
    """Quantize float32 to uint8"""
    return np.clip(np.round(data / scale) - offset, 0, 255).astype(np.uint8)

def dequantize(data, scale, offset):
    """Dequantize uint8 to float32"""
    return (data.astype(np.float32) + offset) * scale

def transpose_onnx_to_qnn(data, onnx_shape, qnn_shape):
    """Transpose from ONNX format to QNN format"""
    data = data.reshape(onnx_shape)
    # Transpose last two dims: (0, 2, 1) for 3D tensors
    if len(onnx_shape) == 3:
        data = data.transpose(0, 2, 1)
    return data.flatten()

def prepare_text_encoder_inputs():
    """Prepare text encoder inputs from calibration data"""
    print("=== Preparing Text Encoder Inputs ===")
    os.makedirs(INPUT_DIR, exist_ok=True)

    # 1. text_ids (INT32, no quantization)
    text_ids = np.fromfile(f'{CALIB_DIR}/text_encoder/text_ids.raw', dtype=np.int32)
    text_ids.tofile(f'{INPUT_DIR}/text_ids.raw')
    print(f"  text_ids: {text_ids.size} elements, {os.path.getsize(f'{INPUT_DIR}/text_ids.raw')} bytes")

    # 2. style_ttl (transpose + quantize)
    style_ttl = np.fromfile(f'{CALIB_DIR}/text_encoder/style_ttl.raw', dtype=np.float32)
    style_ttl = transpose_onnx_to_qnn(style_ttl, (1, 50, 256), (1, 256, 50))
    style_ttl_q = quantize(style_ttl, 0.0054120589047670, -172)
    style_ttl_q.tofile(f'{INPUT_DIR}/style_ttl.raw')
    print(f"  style_ttl: {style_ttl_q.size} elements, {os.path.getsize(f'{INPUT_DIR}/style_ttl.raw')} bytes")

    # 3. text_mask (transpose + quantize)
    text_mask = np.fromfile(f'{CALIB_DIR}/text_encoder/text_mask.raw', dtype=np.float32)
    text_mask = transpose_onnx_to_qnn(text_mask, (1, 1, 128), (1, 128, 1))
    text_mask_q = quantize(text_mask, 0.0039215688593686, 0)
    text_mask_q.tofile(f'{INPUT_DIR}/text_mask.raw')
    print(f"  text_mask: {text_mask_q.size} elements, {os.path.getsize(f'{INPUT_DIR}/text_mask.raw')} bytes")

    # Create input list
    with open(f'{INPUT_DIR}/text_enc_input.txt', 'w') as f:
        f.write('./text_ids.raw ./style_ttl.raw ./text_mask.raw\n')
    print(f"  Created: {INPUT_DIR}/text_enc_input.txt")

def prepare_vector_estimator_inputs(text_emb_path=None):
    """Prepare vector estimator inputs"""
    print("\n=== Preparing Vector Estimator Inputs ===")

    # 1. noisy_latent (transpose + quantize)
    noisy_latent = np.fromfile(f'{CALIB_DIR}/vector_estimator/noisy_latent.raw', dtype=np.float32)
    noisy_latent = transpose_onnx_to_qnn(noisy_latent, (1, 144, 256), (1, 256, 144))
    noisy_latent_q = quantize(noisy_latent, 0.0322749949991703, -127)
    noisy_latent_q.tofile(f'{INPUT_DIR}/noisy_latent.raw')
    print(f"  noisy_latent: {noisy_latent_q.size} elements, {os.path.getsize(f'{INPUT_DIR}/noisy_latent.raw')} bytes")

    # 2. text_emb (from text_encoder output or calibration)
    if text_emb_path and os.path.exists(text_emb_path):
        text_emb = np.fromfile(text_emb_path, dtype=np.float32)
        print(f"  Using text_emb from: {text_emb_path}")
    else:
        text_emb = np.fromfile(f'{CALIB_DIR}/vector_estimator/text_emb.raw', dtype=np.float32)
        print(f"  Using text_emb from calibration")
    text_emb_q = quantize(text_emb, 0.0278750918805599, -128)
    text_emb_q.tofile(f'{INPUT_DIR}/text_emb.raw')
    print(f"  text_emb: {text_emb_q.size} elements, {os.path.getsize(f'{INPUT_DIR}/text_emb.raw')} bytes")

    # 3. style_ttl (already prepared, same as text_encoder)

    # 4. latent_mask (transpose + quantize)
    latent_mask = np.fromfile(f'{CALIB_DIR}/vector_estimator/latent_mask.raw', dtype=np.float32)
    latent_mask = transpose_onnx_to_qnn(latent_mask, (1, 1, 256), (1, 256, 1))
    latent_mask_q = quantize(latent_mask, 0.0039215688593686, 0)
    latent_mask_q.tofile(f'{INPUT_DIR}/latent_mask.raw')
    print(f"  latent_mask: {latent_mask_q.size} elements, {os.path.getsize(f'{INPUT_DIR}/latent_mask.raw')} bytes")

    # 5. text_mask (already prepared)

    # 6. current_step and total_step
    current_step_q = np.array([0], dtype=np.uint8)
    current_step_q.tofile(f'{INPUT_DIR}/current_step.raw')

    total_step_q = quantize(np.array([10], dtype=np.float32), 0.0196078438311815, 0)
    total_step_q.tofile(f'{INPUT_DIR}/total_step.raw')
    print(f"  current_step: {os.path.getsize(f'{INPUT_DIR}/current_step.raw')} bytes")
    print(f"  total_step: {os.path.getsize(f'{INPUT_DIR}/total_step.raw')} bytes")

    # Create input list
    with open(f'{INPUT_DIR}/vec_est_input.txt', 'w') as f:
        f.write('./noisy_latent.raw ./text_emb.raw ./style_ttl.raw ./latent_mask.raw ./text_mask.raw ./current_step.raw ./total_step.raw\n')
    print(f"  Created: {INPUT_DIR}/vec_est_input.txt")

def prepare_vocoder_inputs(latent_path=None):
    """Prepare vocoder inputs"""
    print("\n=== Preparing Vocoder Inputs ===")

    if latent_path and os.path.exists(latent_path):
        latent = np.fromfile(latent_path, dtype=np.float32)
        print(f"  Using latent from: {latent_path}")
    else:
        # Use calibration latent (transpose from ONNX format)
        latent = np.fromfile(f'{CALIB_DIR}/vocoder/latent.raw', dtype=np.float32)
        latent = transpose_onnx_to_qnn(latent, (1, 144, 256), (1, 256, 144))
        print(f"  Using latent from calibration")

    latent_q = quantize(latent, 0.0272593162953854, -129)
    latent_q.tofile(f'{INPUT_DIR}/latent.raw')
    print(f"  latent: {latent_q.size} elements, {os.path.getsize(f'{INPUT_DIR}/latent.raw')} bytes")

    # Create input list
    with open(f'{INPUT_DIR}/vocoder_input.txt', 'w') as f:
        f.write('./latent.raw\n')
    print(f"  Created: {INPUT_DIR}/vocoder_input.txt")

def prepare_all_inputs():
    """Prepare all inputs from calibration data"""
    prepare_text_encoder_inputs()
    prepare_vector_estimator_inputs()
    prepare_vocoder_inputs()
    print("\n=== All inputs prepared in ./inputs/ ===")

def convert_output_to_wav(raw_path, wav_path, sample_rate=44100):
    """Convert raw audio output to WAV"""
    from scipy.io import wavfile

    audio = np.fromfile(raw_path, dtype=np.float32)
    wavfile.write(wav_path, sample_rate, audio)
    print(f"Saved: {wav_path} ({audio.size} samples, {audio.size/sample_rate:.2f} sec)")

def generate_board_script():
    """Generate the inference script to run on the board"""
    script = '''#!/bin/bash
# TTS Inference Script for QCS6490
# Run from directory containing models and inputs

set -e

MODEL_DIR="."
INPUT_DIR="."
OUTPUT_DIR="./output"
NUM_STEPS=10

mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "TTS Inference Pipeline - QCS6490"
echo "=============================================="

# Step 1: Text Encoder
echo ""
echo "[1/3] Running Text Encoder..."
qnn-net-run --model ${MODEL_DIR}/libtext_encoder_htp.so \\
            --backend libQnnHtp.so \\
            --input_list ${INPUT_DIR}/text_enc_input.txt \\
            --output_dir ${OUTPUT_DIR}/text_encoder \\
            --use_native_input_files
cp ${OUTPUT_DIR}/text_encoder/Result_0/text_emb.raw ${INPUT_DIR}/text_emb_out.raw
echo "  Output: text_emb.raw"

# Requantize text_emb for vector estimator
python3 -c "
import numpy as np
data = np.fromfile('${INPUT_DIR}/text_emb_out.raw', dtype=np.float32)
q = np.clip(np.round(data / 0.0278750918805599) + 128, 0, 255).astype(np.uint8)
q.tofile('${INPUT_DIR}/text_emb.raw')
"

# Step 2: Vector Estimator (10 diffusion steps)
echo ""
echo "[2/3] Running Vector Estimator (${NUM_STEPS} steps)..."

for step in $(seq 0 $((NUM_STEPS-1))); do
    echo "  Step $((step+1))/${NUM_STEPS}..."

    # Update current_step
    python3 -c "import numpy as np; np.array([${step}], dtype=np.uint8).tofile('${INPUT_DIR}/current_step.raw')"

    # If step > 0, use previous output as input
    if [ $step -gt 0 ]; then
        python3 -c "
import numpy as np
data = np.fromfile('${OUTPUT_DIR}/vector_estimator/Result_0/denoised_latent.raw', dtype=np.float32)
q = np.clip(np.round(data / 0.0322749949991703) + 127, 0, 255).astype(np.uint8)
q.tofile('${INPUT_DIR}/noisy_latent.raw')
"
    fi

    qnn-net-run --model ${MODEL_DIR}/libvector_estimator_htp.so \\
                --backend libQnnHtp.so \\
                --input_list ${INPUT_DIR}/vec_est_input.txt \\
                --output_dir ${OUTPUT_DIR}/vector_estimator \\
                --use_native_input_files 2>&1 | grep -v "^qnn-net-run"
done

cp ${OUTPUT_DIR}/vector_estimator/Result_0/denoised_latent.raw ${INPUT_DIR}/latent_out.raw
echo "  Output: denoised_latent.raw"

# Requantize latent for vocoder
python3 -c "
import numpy as np
data = np.fromfile('${INPUT_DIR}/latent_out.raw', dtype=np.float32)
q = np.clip(np.round(data / 0.0272593162953854) + 129, 0, 255).astype(np.uint8)
q.tofile('${INPUT_DIR}/latent.raw')
"

# Step 3: Vocoder
echo ""
echo "[3/3] Running Vocoder..."
qnn-net-run --model ${MODEL_DIR}/libvocoder_htp.so \\
            --backend libQnnHtp.so \\
            --input_list ${INPUT_DIR}/vocoder_input.txt \\
            --output_dir ${OUTPUT_DIR}/vocoder \\
            --use_native_input_files

cp ${OUTPUT_DIR}/vocoder/Result_0/wav_tts.raw ${OUTPUT_DIR}/wav_tts.raw
echo "  Output: wav_tts.raw"

echo ""
echo "=============================================="
echo "Inference Complete!"
echo "=============================================="
echo "Output: ${OUTPUT_DIR}/wav_tts.raw"
echo "  Size: $(stat -c %s ${OUTPUT_DIR}/wav_tts.raw) bytes"
'''

    with open(f'{INPUT_DIR}/run_inference.sh', 'w', newline='\n') as f:
        f.write(script)
    print(f"Generated: {INPUT_DIR}/run_inference.sh")

def main():
    parser = argparse.ArgumentParser(description='TTS Inference Pipeline')
    parser.add_argument('--prepare', action='store_true', help='Prepare inputs from calibration')
    parser.add_argument('--script', action='store_true', help='Generate board inference script')
    parser.add_argument('--convert', type=str, help='Convert raw audio to WAV')
    parser.add_argument('--output', type=str, default='output.wav', help='Output WAV filename')

    args = parser.parse_args()

    if args.prepare:
        prepare_all_inputs()
        generate_board_script()
    elif args.script:
        generate_board_script()
    elif args.convert:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        convert_output_to_wav(args.convert, f'{OUTPUT_DIR}/{args.output}')
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
