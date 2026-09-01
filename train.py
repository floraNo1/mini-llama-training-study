"""Train a small randomly initialized Llama-style causal language model."""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tiny_llama.json")
    parser.add_argument("--dataset", default="noanabeshima/TinyStoriesV2")
    parser.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--train-split", default="train[:1%]")
    parser.add_argument("--validation-split", default="validation[:500]")
    parser.add_argument("--output-dir", default="outputs/tiny-llama")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def tokenize_batch(examples, tokenizer, max_length):
    encoded = tokenizer(
        examples["text"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 1,
    )
    input_ids = [tokens + [tokenizer.eos_token_id] for tokens in encoded["input_ids"]]
    return {
        "input_ids": input_ids,
        "attention_mask": [[1] * len(tokens) for tokens in input_ids],
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.smoke_test:
        args.train_split = "train[:32]"
        args.validation_split = "validation[:16]"
        args.max_steps = 1
        args.max_length = min(args.max_length, 128)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig.from_json_file(args.config)
    model = LlamaForCausalLM(config)

    train_dataset = load_dataset(args.dataset, split=args.train_split)
    validation_dataset = load_dataset(args.dataset, split=args.validation_split)
    num_proc = max(1, min(4, os.cpu_count() or 1))

    def preprocess(examples):
        return tokenize_batch(examples, tokenizer, args.max_length)

    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        num_proc=num_proc,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training split",
    )
    validation_dataset = validation_dataset.map(
        preprocess,
        batched=True,
        num_proc=num_proc,
        remove_columns=validation_dataset.column_names,
        desc="Tokenizing validation split",
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
