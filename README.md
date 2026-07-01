![Model](https://img.shields.io/badge/Model-Qwen2.5--0.5B-blue)
![Framework](https://img.shields.io/badge/Framework-Unsloth-green)
![Stage](https://img.shields.io/badge/Pipeline-3--Stage-orange)

<p align="left">
  <img src="https://img.shields.io/badge/Model-Qwen2.5--0.5B-blue" height="50">
  <img src="https://img.shields.io/badge/Framework-Unsloth-green" height="50">
  <img src="https://img.shields.io/badge/Pipeline-3--Stage-orange" height="50">
</p>

# Domain-Specific AI Assistant Fine-Tuning — E-commerce Product Support

## 1. Project Title

**E-commerce Product Support Assistant** — a domain-specific LLM fine-tuned with
[Unsloth](https://github.com/unslothai/unsloth) through a 3-stage pipeline: non-instruction
fine-tuning, instruction fine-tuning, and DPO preference alignment.

## 2. Domain Selected

E-commerce customer/product support: order cancellations, returns, delivery tracking,
gift cards, damaged packages, electronics/firmware issues, marketplace seller shipping,
refund processing, account recovery, and subscription renewals.

## 3. Business Problem

As a GenAI Engineer, the task was to build an internal AI assistant that understands
e-commerce support terminology and answers customer/agent questions clearly — with
**better, more specific, more domain-grounded answers than a generic base model**, which
tends to give plausible-sounding but vague or occasionally incorrect responses (e.g.
incorrectly implying gift cards can be exchanged for cash).

## 4. Dataset Details

| File | Description | Size |
|---|---|---|
| `data/non_instruction_data.txt` | Raw, unstructured e-commerce support text (extracted from a raw support-notes PDF) covering 10 recurring topics, repeated with realistic noise (inconsistent punctuation/spacing, boilerplate internal notes) | ~21K characters |
| `data/instruction_dataset.jsonl` | Instruction → input → output pairs for SFT, covering the same 10 topics, 10 examples each | 100 examples |
| `data/preference_dataset.jsonl` | Prompt / chosen / rejected triples for DPO — chosen responses are correct, helpful, safe, professional, domain-specific; rejected responses are wrong, incomplete, unsafe, rude, generic, or off-domain | 100 examples |

## 5. Base Model Used

`unsloth/Qwen2.5-0.5B-bnb-4bit` — a small, fast, Unsloth-optimized 4-bit model, chosen
from the recommended list (TinyLlama-1.1B, Qwen2.5-0.5B/1.5B, Llama-3.2-1B, Gemma small)
so the full 3-stage pipeline can run on a free Colab T4 GPU.

## 6. Non-Instruction Fine-Tuning Approach (Stage 1)

- Raw text is cleaned (whitespace/punctuation normalization) and split into per-topic
  paragraph chunks, removing repeated "Internal note N" boilerplate.
- Trained with `SFTTrainer` using `dataset_text_field="text"`, `packing=True` (short raw
  paragraphs are packed into full sequences), no instruction formatting.
- Goal: shift the model toward e-commerce support vocabulary and facts before it learns
  how to answer questions.
- Notebook: `notebooks/non_instruction_finetuning.ipynb`

## 7. Instruction Fine-Tuning Approach (Stage 2)

- Continues from the Stage 1 merged model.
- Formats each example into an `### Instruction / ### Input / ### Response` prompt
  template using `data/instruction_dataset.jsonl`.
- Trained with `SFTTrainer`, `packing=False` (to preserve prompt/response boundaries).
- Evaluated on the same 10 held-out questions used for the base model.
- Notebook: `notebooks/instruction_finetuning.ipynb`

## 8. DPO Alignment Approach (Stage 3)

- Continues from the Stage 2 merged model.
- Loads `data/preference_dataset.jsonl` (prompt/chosen/rejected, already formatted in
  the same `### Instruction / ### Response` style).
- Trained with `DPOTrainer` (`beta=0.1`), no separate reference model needed since
  Unsloth/PEFT computes the reference log-probs from the frozen base weights.
- Goal: keep Stage 2's domain knowledge but make the model consistently prefer correct,
  helpful, safe, professional, domain-specific responses.
- Notebook: `notebooks/dpo_alignment.ipynb`

## 9. LoRA / QLoRA Configuration

| Stage | Rank | Alpha | Dropout | LR | Batch size (effective) | Max steps |
|---|---|---|---|---|---|---|
| 1 — Non-instruction | 16 | 16 | 0.0 | 2e-4 | 8 (2 × grad-accum 4) | 60 |
| 2 — Instruction SFT | 16 | 16 | 0.0 | 1e-4 | 8 (2 × grad-accum 4) | 90 |
| 3 — DPO | 16 | 16 | 0.0 | 5e-6 | 8 (2 × grad-accum 4) | 60 |

Base model is loaded in 4-bit (QLoRA-style) via `load_in_4bit=True`, target modules:
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

See `reports/fine_tuning_explanation.md` for the reasoning behind each setting.

## 10. Training Screenshots or Logs

> Add screenshots of your Colab training run output here (loss curves, `trainer.train()`
> logs, GPU memory stats) after you execute the notebooks. Suggested location:
> `reports/screenshots/`.

## 11. Before vs After Output Comparison

See:
- `reports/base_model_evaluation.md` — base model on 10 domain questions
- `reports/sft_model_comparison.md` — base vs SFT model
- `reports/final_evaluation.md` — base vs SFT vs DPO model, all 10 questions

## 12. Final Observations

- The base model is fluent but generic, and not grounded in the company's specific
  support policies — it occasionally gives information that doesn't match our actual
  process (e.g. gift card cash exchange).
- Non-instruction fine-tuning shifts the model's "vocabulary" toward the domain but does
  not, by itself, teach question-answering behavior.
- Instruction fine-tuning is what makes the model actually answer questions helpfully
  and with the right facts.
- DPO alignment is a smaller but meaningful final step: it does not add new facts, but
  it makes responses more consistently correct, confident, and professionally toned by
  directly optimizing against rejected (generic/unsafe/rude) response patterns.

## 13. Challenges Faced

- Raw support text required noticeable cleaning (inconsistent punctuation/spacing,
  repeated boilerplate notes) before it was usable for next-token training.
- Keeping prompt formatting **identical** across the instruction and preference datasets
  was necessary so the DPO stage could be trained directly on top of the SFT model
  without a format mismatch.
- Small models (0.5B) have limited capacity, so steps/LR were kept modest to avoid
  overfitting on a 100-example dataset.

## 14. Future Improvements

- Expand the instruction and preference datasets beyond 100 examples each, including
  harder edge cases and multi-turn conversations.
- Try a slightly larger base model (Qwen2.5-1.5B or Llama-3.2-1B) for richer domain
  reasoning, if GPU budget allows.
- Add automated evaluation (e.g. LLM-as-judge scoring) instead of manual before/after
  tables.
- Experiment with ORPO as a single-stage alternative to SFT + DPO.

---

## Repository Structure

```
domain-ai-assistant-finetuning/
│
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
│
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
│
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
│
├── src/
│   └── inference.py
│
├── README.md
└── requirements.txt
```

## How to Run

1. Open each notebook in Google Colab (GPU runtime, e.g. T4).
2. Upload the corresponding file(s) from `data/` to `/content/` when prompted.
3. Run `notebooks/non_instruction_finetuning.ipynb` → produces Stage 1 adapter/model.
4. Run `notebooks/instruction_finetuning.ipynb` → produces Stage 2 adapter/model.
5. Run `notebooks/dpo_alignment.ipynb` → produces the final DPO-aligned model.
6. Push the final model to Hugging Face Hub (uncomment the `push_to_hub_merged` line in
   each notebook, using your own HF token) under your account, e.g.
   `shahmitul1809/ecommerce-support-dpo`.
7. Use `src/inference.py` to query the final model:
   ```bash
   python src/inference.py --question "How can I apply for reimbursement?"
   ```
