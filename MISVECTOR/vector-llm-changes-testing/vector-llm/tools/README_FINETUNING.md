# Vector LLM Fine-Tuning Guide

This guide details how to fine-tune a custom local model (Qwen 2.5 7B / Llama 3.1 8B) for the Vector microservice and register it in Ollama.

---

## Step 1: Generate Fine-Tuning Dataset

Run `tools/dataset_builder.py` to generate synthetic training pairs based on your application's domain queries, Plotly specs, and decision outputs:

```bash
python tools/dataset_builder.py --output_dir ./tools/data --num_samples 500
```

This creates `./tools/data/vector_llm_train.jsonl` in Alpaca format.

---

## Step 2: Fine-Tune with Unsloth & QLoRA

Run the fine-tuning script on a machine with an NVIDIA GPU (or Google Colab / AWS T4):

```bash
python tools/finetune_unsloth.py --data_path ./tools/data/vector_llm_train.jsonl --output_dir ./tools/models/vector_llm_lora
```

- Uses 4-bit QLoRA to keep VRAM requirements under 8GB-12GB.
- Automatically saves merged weights and exports `.gguf` (`q4_k_m`) model file into `./tools/models/vector_llm_lora/gguf/`.

---

## Step 3: Register Model in Local Ollama Server

Navigate to the `tools` directory and run:

```bash
ollama create vector-llm:v1 -f Modelfile
```

Verify that the model is loaded:

```bash
ollama list
```

You should see `vector-llm:v1` in the list of available models.

---

## Step 4: Update Microservice Config

Edit `.env` or `config.py` to use your newly registered model:

```env
OLLAMA_MODEL=vector-llm:v1
```

Restart the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Your microservice will now serve responses using your custom fine-tuned weights!
