#!/usr/bin/env python3
"""
Prepare input files for QNN TTS inference pipeline.
Handles transposition from ONNX format to QNN model format.

Quantization parameters (from cpp files):
- style_ttl:    scale=0.0054120589, offset=-172
- style_dp:     scale=0.0054351119, offset=-161
- text_mask:    scale=0.0039215689, offset=0
- latent_mask:  scale=0.0039215689, offset=0
- noisy_latent: scale=0.0322749950, offset=-127
- text_emb:     scale=0.0278750919, offset=-128
- denoised:     scale=0.0271500126, offset=-118
- wav_tts:      scale=0.0032209943, offset=-134
"""

import numpy as np
import os

INPUT_DIR = "./inputs"
os.makedirs(INPUT_DIR, exist_ok=True)


def quantize(data: np.ndarray, scale: float, offset: int) -> np.ndarray:
    """Quantize float data to UINT8."""
    quantized = np.round(data / scale) - offset
    return np.clip(quantized, 0, 255).astype(np.uint8)


def dequantize(data: np.ndarray, scale: float, offset: int) -> np.ndarray:
    """Dequantize UINT8 data to float."""
    return (data.astype(np.float32) + offset) * scale


def prepare_text_ids(text_ids: np.ndarray) -> None:
    """
    Prepare text_ids input.
    Shape: [1, 128] INT32 - no transpose needed
    """
    assert text_ids.shape == (1, 128), f"Expected [1, 128], got {text_ids.shape}"
    text_ids = text_ids.astype(np.int32)
    text_ids.tofile(os.path.join(INPUT_DIR, "text_ids.raw"))
    print(f"Saved text_ids.raw: {text_ids.shape} INT32 ({text_ids.nbytes} bytes)")


def prepare_style_ttl(style_ttl: np.ndarray) -> None:
    """
    Prepare style_ttl input for text_encoder and vector_estimator.
    ONNX shape: [1, 50, 256] -> Model shape: [1, 256, 50] (transpose axes 1,2)
    """
    assert style_ttl.shape == (1, 50, 256), f"Expected [1, 50, 256], got {style_ttl.shape}"

    # Transpose to model format
    style_ttl_transposed = np.transpose(style_ttl, (0, 2, 1))  # [1, 256, 50]

    # Quantize
    quantized = quantize(style_ttl_transposed, scale=0.0054120589, offset=-172)
    quantized.tofile(os.path.join(INPUT_DIR, "style_ttl.raw"))
    print(f"Saved style_ttl.raw: {quantized.shape} UINT8 ({quantized.nbytes} bytes)")


def prepare_style_dp(style_dp: np.ndarray) -> None:
    """
    Prepare style_dp input for duration_predictor.
    ONNX shape: [1, 8, 16] -> Model shape: [1, 16, 8] (transpose axes 1,2)
    """
    assert style_dp.shape == (1, 8, 16), f"Expected [1, 8, 16], got {style_dp.shape}"

    # Transpose to model format
    style_dp_transposed = np.transpose(style_dp, (0, 2, 1))  # [1, 16, 8]

    # Quantize
    quantized = quantize(style_dp_transposed, scale=0.0054351119, offset=-161)
    quantized.tofile(os.path.join(INPUT_DIR, "style_dp.raw"))
    print(f"Saved style_dp.raw: {quantized.shape} UINT8 ({quantized.nbytes} bytes)")


def prepare_text_mask(text_mask: np.ndarray) -> None:
    """
    Prepare text_mask input.
    ONNX shape: [1, 1, 128] -> Model shape: [1, 128, 1] (transpose axes 1,2)
    """
    assert text_mask.shape == (1, 1, 128), f"Expected [1, 1, 128], got {text_mask.shape}"

    # Transpose to model format
    text_mask_transposed = np.transpose(text_mask, (0, 2, 1))  # [1, 128, 1]

    # Quantize
    quantized = quantize(text_mask_transposed, scale=0.0039215689, offset=0)
    quantized.tofile(os.path.join(INPUT_DIR, "text_mask.raw"))
    print(f"Saved text_mask.raw: {quantized.shape} UINT8 ({quantized.nbytes} bytes)")


def prepare_latent_mask(latent_mask: np.ndarray) -> None:
    """
    Prepare latent_mask input for vector_estimator.
    ONNX shape: [1, 1, 256] -> Model shape: [1, 256, 1] (transpose axes 1,2)
    """
    assert latent_mask.shape == (1, 1, 256), f"Expected [1, 1, 256], got {latent_mask.shape}"

    # Transpose to model format
    latent_mask_transposed = np.transpose(latent_mask, (0, 2, 1))  # [1, 256, 1]

    # Quantize
    quantized = quantize(latent_mask_transposed, scale=0.0039215689, offset=0)
    quantized.tofile(os.path.join(INPUT_DIR, "latent_mask.raw"))
    print(f"Saved latent_mask.raw: {quantized.shape} UINT8 ({quantized.nbytes} bytes)")


def prepare_noisy_latent(noisy_latent: np.ndarray) -> None:
    """
    Prepare noisy_latent input for vector_estimator.
    ONNX shape: [1, 144, 256] -> Model shape: [1, 256, 144] (transpose axes 1,2)
    """
    assert noisy_latent.shape == (1, 144, 256), f"Expected [1, 144, 256], got {noisy_latent.shape}"

    # Transpose to model format
    noisy_latent_transposed = np.transpose(noisy_latent, (0, 2, 1))  # [1, 256, 144]

    # Quantize
    quantized = quantize(noisy_latent_transposed, scale=0.0322749950, offset=-127)
    quantized.tofile(os.path.join(INPUT_DIR, "noisy_latent.raw"))
    print(f"Saved noisy_latent.raw: {quantized.shape} UINT8 ({quantized.nbytes} bytes)")


def dequantize_wav_output(wav_path: str, output_path: str = None) -> np.ndarray:
    """
    Dequantize wav_tts output from vocoder.
    Model shape: [1, 786432] UINT8
    """
    wav_quantized = np.fromfile(wav_path, dtype=np.uint8)
    wav_float = dequantize(wav_quantized, scale=0.0032209943, offset=-134)

    if output_path:
        wav_float.tofile(output_path)
        print(f"Saved dequantized wav to {output_path}")

    return wav_float


def create_dummy_inputs():
    """Create dummy inputs for testing the pipeline."""
    print("Creating dummy inputs for testing...")
    print("-" * 50)

    # text_ids: [1, 128] INT32 - token IDs (0-255 range typically)
    text_ids = np.zeros((1, 128), dtype=np.int32)
    text_ids[0, :10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Dummy tokens
    prepare_text_ids(text_ids)

    # style_ttl: [1, 50, 256] - style embedding
    style_ttl = np.random.randn(1, 50, 256).astype(np.float32) * 0.5
    prepare_style_ttl(style_ttl)

    # style_dp: [1, 8, 16] - duration predictor style
    style_dp = np.random.randn(1, 8, 16).astype(np.float32) * 0.5
    prepare_style_dp(style_dp)

    # text_mask: [1, 1, 128] - mask for valid text tokens (1.0 for valid, 0.0 for padding)
    text_mask = np.zeros((1, 1, 128), dtype=np.float32)
    text_mask[0, 0, :10] = 1.0  # First 10 tokens are valid
    prepare_text_mask(text_mask)

    # latent_mask: [1, 1, 256] - mask for valid latent frames
    latent_mask = np.ones((1, 1, 256), dtype=np.float32)
    prepare_latent_mask(latent_mask)

    # noisy_latent: [1, 144, 256] - initial noisy latent
    noisy_latent = np.random.randn(1, 144, 256).astype(np.float32)
    prepare_noisy_latent(noisy_latent)

    print("-" * 50)
    print("Done! All dummy inputs saved to ./inputs/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare inputs for QNN TTS inference")
    parser.add_argument("--dummy", action="store_true", help="Create dummy inputs for testing")
    parser.add_argument("--dequantize-wav", type=str, help="Path to wav_tts.raw to dequantize")
    parser.add_argument("--output", type=str, help="Output path for dequantized wav")

    args = parser.parse_args()

    if args.dummy:
        create_dummy_inputs()
    elif args.dequantize_wav:
        wav = dequantize_wav_output(args.dequantize_wav, args.output)
        print(f"Dequantized wav shape: {wav.shape}, range: [{wav.min():.4f}, {wav.max():.4f}]")
    else:
        print("Usage:")
        print("  python prepare_inputs.py --dummy              # Create dummy test inputs")
        print("  python prepare_inputs.py --dequantize-wav outputs/wav_tts.raw --output audio.raw")
