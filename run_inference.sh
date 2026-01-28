#!/bin/bash
#
# TTS Inference Pipeline for QCS6490 using QNN
# Models: text_encoder -> duration_predictor -> vector_estimator (N steps) -> vocoder
#

set -e

# Configuration
MODEL_DIR="./QNN_Models"
INPUT_DIR="./inputs"
OUTPUT_DIR="./outputs"
NUM_DIFFUSION_STEPS=10  # Number of denoising steps for vector_estimator

# QNN Runtime settings
BACKEND_LIB="libQnnHtp.so"

# Create directories
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

echo "=============================================="
echo "TTS Inference Pipeline - QCS6490 HTP"
echo "=============================================="

# ---------------------------------------------------------------------------
# Model Input/Output Specifications (from calibration files):
# ALL INPUTS ARE FLOAT32! Model quantizes internally.
# ---------------------------------------------------------------------------
# 1. text_encoder_htp:
#    Inputs:
#      - text_ids:   [1, 128]      INT32   (512 bytes)
#      - style_ttl:  [1, 50, 256]  FLOAT32 (51200 bytes) - NO transpose needed
#      - text_mask:  [1, 1, 128]   FLOAT32 (512 bytes)   - NO transpose needed
#    Output:
#      - text_emb:   [1, 256, 128] FLOAT32
#
# 2. duration_predictor_htp:
#    Inputs:
#      - text_ids:   [1, 128]      INT32   (512 bytes)
#      - style_dp:   [1, 8, 16]    FLOAT32 (512 bytes)   - NO transpose needed
#      - text_mask:  [1, 1, 128]   FLOAT32 (512 bytes)   - NO transpose needed
#    Output:
#      - duration:   [1]           FLOAT32
#
# 3. vector_estimator_htp:
#    Inputs:
#      - noisy_latent: [1, 144, 256] FLOAT32 (147456 bytes) - NO transpose needed
#      - text_emb:     [1, 256, 128] FLOAT32 (131072 bytes) - from text_encoder
#      - style_ttl:    [1, 50, 256]  FLOAT32 (51200 bytes)  - NO transpose needed
#      - latent_mask:  [1, 1, 256]   FLOAT32 (1024 bytes)   - NO transpose needed
#      - text_mask:    [1, 1, 128]   FLOAT32 (512 bytes)    - NO transpose needed
#      - current_step: [1]           FLOAT32 (4 bytes)
#      - total_step:   [1]           FLOAT32 (4 bytes)
#    Output:
#      - denoised_latent: [1, 144, 256] FLOAT32
#
# 4. vocoder_htp:
#    Inputs:
#      - latent:     [1, 144, 256] FLOAT32 (147456 bytes) - NO transpose needed
#    Output:
#      - wav_tts:    [1, 786432]   FLOAT32
# ---------------------------------------------------------------------------

# Helper function to create input list files for qnn-net-run
# All inputs for one inference must be on a SINGLE line, space-separated
# Each line = one inference batch
create_input_list() {
    local model_name=$1
    local input_list_file="$INPUT_DIR/${model_name}_input_list.txt"
    shift
    # Write all input files on a single line, space-separated
    echo "$*" > "$input_list_file"
    echo "$input_list_file"
}

# ---------------------------------------------------------------------------
# Step 1: Run Text Encoder
# ---------------------------------------------------------------------------
run_text_encoder() {
    echo ""
    echo "[Step 1/4] Running Text Encoder..."

    # Input files (must be pre-transposed to model format):
    # - text_ids.raw:   [1, 128]     INT32  (512 bytes)
    # - style_ttl.raw:  [1, 256, 50] UINT8  (12800 bytes) - transposed from [1, 50, 256]
    # - text_mask.raw:  [1, 128, 1]  UINT8  (128 bytes)   - transposed from [1, 1, 128]

    local input_list=$(create_input_list "text_encoder" \
        "${INPUT_DIR}/text_ids.raw" \
        "${INPUT_DIR}/style_ttl.raw" \
        "${INPUT_DIR}/text_mask.raw")

    qnn-net-run \
        --model "${MODEL_DIR}/libtext_encoder_htp.so" \
        --backend "$BACKEND_LIB" \
        --input_list "$input_list" \
        --output_dir "${OUTPUT_DIR}/text_encoder"

    # Output: text_emb.raw [1, 128, 256] UINT8 (32768 bytes)
    cp "${OUTPUT_DIR}/text_encoder/Result_0/text_emb.raw" "${OUTPUT_DIR}/text_emb.raw"
    echo "  -> Output: text_emb.raw [1, 128, 256]"
}

# ---------------------------------------------------------------------------
# Step 2: Run Duration Predictor
# ---------------------------------------------------------------------------
run_duration_predictor() {
    echo ""
    echo "[Step 2/4] Running Duration Predictor..."

    # Input files:
    # - text_ids.raw:  [1, 128]     INT32  (512 bytes)
    # - style_dp.raw:  [1, 16, 8]   UINT8  (128 bytes)   - transposed from [1, 8, 16]
    # - text_mask.raw: [1, 128, 1]  UINT8  (128 bytes)   - transposed from [1, 1, 128]

    local input_list=$(create_input_list "duration_predictor" \
        "${INPUT_DIR}/text_ids.raw" \
        "${INPUT_DIR}/style_dp.raw" \
        "${INPUT_DIR}/text_mask.raw")

    qnn-net-run \
        --model "${MODEL_DIR}/libduration_predictor_htp.so" \
        --backend "$BACKEND_LIB" \
        --input_list "$input_list" \
        --output_dir "${OUTPUT_DIR}/duration_predictor"

    # Output: duration.raw [1] UINT8 (1 byte)
    cp "${OUTPUT_DIR}/duration_predictor/Result_0/duration.raw" "${OUTPUT_DIR}/duration.raw"
    echo "  -> Output: duration.raw [1]"
}

# ---------------------------------------------------------------------------
# Step 3: Run Vector Estimator (Diffusion Denoising Loop)
# ---------------------------------------------------------------------------
run_vector_estimator() {
    echo ""
    echo "[Step 3/4] Running Vector Estimator (${NUM_DIFFUSION_STEPS} denoising steps)..."

    # Initialize noisy_latent (you need to generate this from noise + duration)
    # For now, assume it's provided in inputs/noisy_latent.raw
    cp "${INPUT_DIR}/noisy_latent.raw" "${OUTPUT_DIR}/current_latent.raw"

    # Create total_step file (FLOAT32 - 4 bytes)
    python3 -c "import numpy as np; np.array([${NUM_DIFFUSION_STEPS}], dtype=np.float32).tofile('${OUTPUT_DIR}/total_step.raw')"

    for ((step=0; step<NUM_DIFFUSION_STEPS; step++)); do
        echo "  Denoising step $((step+1))/${NUM_DIFFUSION_STEPS}..."

        # Create current_step file (FLOAT32 - 4 bytes)
        python3 -c "import numpy as np; np.array([${step}], dtype=np.float32).tofile('${OUTPUT_DIR}/current_step.raw')"

        # Input files for vector_estimator:
        # - noisy_latent:  [1, 256, 144] UINT8 (36864 bytes) - transposed from [1, 144, 256]
        # - text_emb:      [1, 128, 256] UINT8 (32768 bytes) - from text_encoder (already transposed)
        # - style_ttl:     [1, 256, 50]  UINT8 (12800 bytes) - transposed from [1, 50, 256]
        # - latent_mask:   [1, 256, 1]   UINT8 (256 bytes)   - transposed from [1, 1, 256]
        # - text_mask:     [1, 128, 1]   UINT8 (128 bytes)   - transposed from [1, 1, 128]
        # - current_step:  [1]           UINT8 (1 byte)
        # - total_step:    [1]           UINT8 (1 byte)

        local input_list=$(create_input_list "vector_estimator_step${step}" \
            "${OUTPUT_DIR}/current_latent.raw" \
            "${OUTPUT_DIR}/text_emb.raw" \
            "${INPUT_DIR}/style_ttl.raw" \
            "${INPUT_DIR}/latent_mask.raw" \
            "${INPUT_DIR}/text_mask.raw" \
            "${OUTPUT_DIR}/current_step.raw" \
            "${OUTPUT_DIR}/total_step.raw")

        qnn-net-run \
            --model "${MODEL_DIR}/libvector_estimator_htp.so" \
            --backend "$BACKEND_LIB" \
            --input_list "$input_list" \
            --output_dir "${OUTPUT_DIR}/vector_estimator_step${step}"

        # Use denoised output as input for next step
        cp "${OUTPUT_DIR}/vector_estimator_step${step}/Result_0/denoised_latent.raw" \
           "${OUTPUT_DIR}/current_latent.raw"
    done

    # Final denoised latent
    cp "${OUTPUT_DIR}/current_latent.raw" "${OUTPUT_DIR}/denoised_latent.raw"
    echo "  -> Output: denoised_latent.raw [1, 256, 144]"
}

# ---------------------------------------------------------------------------
# Step 4: Run Vocoder
# ---------------------------------------------------------------------------
run_vocoder() {
    echo ""
    echo "[Step 4/4] Running Vocoder..."

    # Input files:
    # - latent: [1, 256, 144] UINT8 (36864 bytes) - from vector_estimator
    #   Note: vocoder expects [1, 256, 144] which is already in model format

    local input_list=$(create_input_list "vocoder" \
        "${OUTPUT_DIR}/denoised_latent.raw")

    qnn-net-run \
        --model "${MODEL_DIR}/libvocoder_htp.so" \
        --backend "$BACKEND_LIB" \
        --input_list "$input_list" \
        --output_dir "${OUTPUT_DIR}/vocoder"

    # Output: wav_tts.raw [1, 786432] UINT8
    cp "${OUTPUT_DIR}/vocoder/Result_0/wav_tts.raw" "${OUTPUT_DIR}/wav_tts.raw"
    echo "  -> Output: wav_tts.raw [1, 786432]"
}

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo "Checking for required input files..."

    required_files=(
        "${INPUT_DIR}/text_ids.raw"
        "${INPUT_DIR}/style_ttl.raw"
        "${INPUT_DIR}/style_dp.raw"
        "${INPUT_DIR}/text_mask.raw"
        "${INPUT_DIR}/latent_mask.raw"
        "${INPUT_DIR}/noisy_latent.raw"
    )

    missing_files=0
    for f in "${required_files[@]}"; do
        if [[ ! -f "$f" ]]; then
            echo "  [MISSING] $f"
            missing_files=1
        else
            echo "  [OK] $f"
        fi
    done

    if [[ $missing_files -eq 1 ]]; then
        echo ""
        echo "ERROR: Missing required input files. See above for details."
        echo ""
        echo "Required input file formats (already transposed to model format):"
        echo "  - text_ids.raw:     [1, 128]      INT32  (512 bytes)"
        echo "  - style_ttl.raw:    [1, 256, 50]  UINT8  (12800 bytes)"
        echo "  - style_dp.raw:     [1, 16, 8]    UINT8  (128 bytes)"
        echo "  - text_mask.raw:    [1, 128, 1]   UINT8  (128 bytes)"
        echo "  - latent_mask.raw:  [1, 256, 1]   UINT8  (256 bytes)"
        echo "  - noisy_latent.raw: [1, 256, 144] UINT8  (36864 bytes)"
        exit 1
    fi

    # Run the pipeline
    run_text_encoder
    run_duration_predictor
    run_vector_estimator
    run_vocoder

    echo ""
    echo "=============================================="
    echo "Inference Complete!"
    echo "=============================================="
    echo "Final output: ${OUTPUT_DIR}/wav_tts.raw"
    echo "  Format: [1, 786432] UINT8 quantized audio samples"
    echo ""
}

# Run main
main "$@"