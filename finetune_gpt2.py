import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")  # gpt2, gpt2-medium, gpt2-large
    p.add_argument("--train_file", default=None, help="Path to a .txt file")
    p.add_argument("--output_dir", default="./gpt2-finetuned")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # ---- Load data ----
    if args.train_file:
        raw = load_dataset("text", data_files={"train": args.train_file})
        split = raw["train"].train_test_split(test_size=0.1, seed=args.seed)
        raw = {"train": split["train"], "validation": split["test"]}
    else:
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    # ---- Tokenize ----
    def tokenize(batch):
        return tokenizer(batch["text"])

    tokenized = {
        k: v.map(tokenize, batched=True, remove_columns=v.column_names)
        for k, v in raw.items()
    }

    # ---- Pack tokens into fixed-size blocks ----
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = (len(concatenated["input_ids"]) // args.block_size) * args.block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total, args.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_data = {k: v.map(group_texts, batched=True) for k, v in tokenized.items()}

    # ---- Train ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        fp16=True,  # set False if no NVIDIA GPU (or use bf16=True on Ampere+)
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_data["train"],
        eval_dataset=lm_data["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ---- Quick test generation ----
    generator = pipeline("text-generation", model=args.output_dir, tokenizer=args.output_dir)
    out = generator("Once upon a time", max_new_tokens=60, do_sample=True, top_p=0.95, temperature=0.8)
    print(out[0]["generated_text"])


if __name__ == "__main__":
    main()