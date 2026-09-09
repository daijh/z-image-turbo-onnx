# Z-Image-Turbo ONNX Exporters

Export the [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) text-to-image pipeline to ONNX models.

`export_models.py` serves as the primary entry point for converting all pipeline components.

## Supported Components

The export script covers all essential parts of the Z-Image-Turbo pipeline:

* **Tokenizer:** Essential JSON and text configurations for prompt processing.
* **Text encoder (Qwen3):** Processes text prompts into embeddings.
* **Transformer trunk:** The core image generation model.
* **VAE decoder:** Converts latent representations back into pixel-space images.
* **Helper models:** Includes `scheduler_step`, `vae_pre_process`, and `sc_prep`.
* **Safety checker:** *(Optional)* Filters out unsafe content. Requires a separate checkpoint (see Usage below).

*Note: Running `export_models.py` builds the full pipeline into a single, self-contained bundle directory.*

---

## Installation

Python 3.13 is recommended.

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Download Checkpoints
Download the model checkpoints locally before exporting. You can do this quickly using the `huggingface_hub` Python library:

```python
from huggingface_hub import snapshot_download

# Download the core Z-Image-Turbo checkpoint
snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir="checkpoints/z-image-turbo")

# Optional: Download the Safety Checker checkpoint
snapshot_download("CompVis/stable-diffusion-safety-checker", local_dir="checkpoints/safety-checker")
```

### 2. Export the Pipeline
By default, the script exports all components to a `z-image-turbo-onnx/` directory.

**Export everything at once:**
```bash
python export_models.py checkpoints/z-image-turbo
```

**Export with the Safety Checker:**
Because the safety checker uses a separate CompVis model, it requires the `--safety_checker_checkpoint` flag. If this flag is provided, the standard full export automatically includes it. If omitted, the safety checker is safely skipped.
```bash
python export_models.py checkpoints/z-image-turbo --safety_checker_checkpoint checkpoints/safety-checker
```

---

## Output Structure

All generated `.onnx` models and their associated external data are written to `<output_dir>/onnx/`. To ensure the output bundle is fully self-contained, `export_models.py` automatically copies the required tokenizer files from the source checkpoint into `<output_dir>/tokenizer/`.

The final output directory (`z-image-turbo-onnx/` by default) will look like this:

```text
z-image-turbo-onnx/
├── onnx/
│   ├── sc_prep_model_f16.onnx
│   ├── scheduler_step_model_f16.onnx
│   ├── text_encoder_model_q4f16.onnx
│   ├── text_encoder_model_q4f16.onnx_data
│   ├── text_encoder_model_q4f16.onnx_data_1
│   ├── transformer_model_q4f16.onnx
│   ├── transformer_model_q4f16.onnx_data
│   ├── transformer_model_q4f16.onnx_data_1
│   ├── vae_decoder_model_f16.onnx
│   └── vae_pre_process_model_f16.onnx
└── tokenizer/
    ├── merges.txt
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.json
```