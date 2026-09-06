"""
tools/finetune_unsloth.py
========================
Fine-tunes Qwen 2.5 7B / Llama 3.1 8B using Unsloth & QLoRA for fast training
and exports the fine-tuned model directly to GGUF format for Ollama.

REQUIREMENTS:
  pip install unsloth torch trl datasets transformers

USAGE:
  python tools/finetune_unsloth.py --data_path ./tools/data/vector_llm_train.jsonl --output_dir ./tools/models/vector-llm-gguf
"""

import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fine-tune local LLM with Unsloth QLoRA")
    parser.add_argument("--base_model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", help="Base model identifier")
    parser.add_argument("--data_path", default="./tools/data/vector_llm_train.jsonl", help="Training JSONL dataset")
    parser.add_argument("--output_dir", default="./tools/models/vector_llm_lora", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    args = parser.parse_args()

    print(f"=== Vector LLM Fine-Tuning Pipeline ===")
    print(f"Base Model: {args.base_model}")
    print(f"Dataset Path: {args.data_path}")
    print(f"Output Directory: {args.output_dir}")

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
    except ImportError:
        print("\n[NOTE] Unsloth / PyTorch dependencies are not installed in the current environment.")
        print("To run fine-tuning on GPU, install:")
        print("  pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
        print("  pip install --no-deps trl peft accelerate bitsandbytes\n")
        print("Refer to tools/README_FINETUNING.md for step-by-step instructions.")
        return

    # 1. Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model,
        max_seq_length = args.max_seq_length,
        load_in_4bit = True,
    )

    # 2. Add LoRA target modules
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # 3. Load dataset
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    def format_prompts(batch):
        instructions = batch["instruction"]
        inputs       = batch["input"]
        outputs      = batch["output"]
        texts = []
        for inst, inp, out in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            texts.append(text)
        return { "text" : texts }

    formatted_dataset = dataset.map(format_prompts, batched=True)

    # 4. Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = args.epochs,
            learning_rate = 2e-4,
            fp16 = True,
            logging_steps = 1,
            output_dir = args.output_dir,
        ),
    )

    # 5. Train
    print("[finetune] Starting training...")
    trainer.train()

    # 6. Save & Export to GGUF (q4_k_m)
    gguf_dir = os.path.join(args.output_dir, "gguf")
    print(f"[finetune] Saving GGUF model to {gguf_dir}...")
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
    print("[finetune] Training and GGUF export complete!")

if __name__ == "__main__":
    main()
