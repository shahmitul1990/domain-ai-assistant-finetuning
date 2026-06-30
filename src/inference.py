"""
src/inference.py

Simple inference script for the final DPO-aligned E-commerce Product Support Assistant.

Usage:
    python src/inference.py
    python src/inference.py --question "How can I apply for reimbursement?"
    python src/inference.py --model_path /content/stage3_dpo_final_merged_model
"""

import argparse

PROMPT_TEMPLATE = "### Instruction:\n{question}\n\n### Response:\n"

DEFAULT_MODEL_PATH = "shahmitul1809/ecommerce-support-dpo"  # Hugging Face Hub repo
# Or, if running locally / in the same Colab session right after training:
# DEFAULT_MODEL_PATH = "/content/stage3_dpo_final_merged_model"

MAX_SEQ_LENGTH = 1024


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load the final DPO-aligned model using Unsloth's FastLanguageModel."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_answer(question: str, model=None, tokenizer=None, model_path: str = DEFAULT_MODEL_PATH,
                     max_new_tokens: int = 150) -> str:
    """Generate an answer to a customer-support question using the fine-tuned model."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)

    prompt = PROMPT_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()


def main():
    parser = argparse.ArgumentParser(description="E-commerce Support Assistant inference")
    parser.add_argument("--question", type=str,
                         default="How can I apply for reimbursement?",
                         help="Customer support question to ask the model")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                         help="Path or Hugging Face repo id of the final DPO-aligned model")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    answer = generate_answer(
        args.question,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    print(answer)


if __name__ == "__main__":
    main()
