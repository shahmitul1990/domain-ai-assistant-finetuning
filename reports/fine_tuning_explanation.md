# Fine-Tuning Concepts Explained

## 1. Why full fine-tuning is expensive

Full fine-tuning updates **every parameter** in the model. For even a "small" model
like Qwen2.5-0.5B that's 500 million parameters, and for larger models it can be tens
of billions. This requires storing full-precision weights, gradients, and optimizer
states (e.g. Adam keeps two extra moving-average tensors per parameter) all in GPU
memory simultaneously — easily 10–20x the model's raw size in VRAM. It also takes much
longer to train and risks **catastrophic forgetting**, where the model loses its
general-purpose abilities while specializing on the new domain. For most teams without
multi-GPU clusters, full fine-tuning of anything beyond a few hundred million parameters
is simply not feasible on a single consumer or Colab GPU.

## 2. What LoRA does

LoRA (Low-Rank Adaptation) freezes the original model weights and instead injects small,
trainable **low-rank matrices** into specific layers (typically the attention projection
layers `q_proj`, `k_proj`, `v_proj`, `o_proj` and the MLP layers `gate_proj`, `up_proj`,
`down_proj`). Instead of learning a full-size weight update `ΔW`, LoRA learns two much
smaller matrices `A` and `B` such that `ΔW ≈ A·B`, where the rank `r` of these matrices
is tiny (e.g. 16) compared to the original weight matrix dimensions. Only `A` and `B` are
trained and saved, which is typically **less than 1% of the original parameter count**,
drastically cutting memory and storage requirements while still achieving strong
task-adaptation performance.

## 3. What QLoRA does

QLoRA combines LoRA with **4-bit quantization** of the frozen base model weights. The
base model is loaded in 4-bit precision (using techniques like NF4 quantization), which
cuts its memory footprint roughly 4x compared to 16-bit weights, while the small LoRA
adapter matrices are still trained in higher precision (e.g. bf16/fp16) on top. Unsloth
implements this efficiently with custom CUDA kernels, so QLoRA is both memory-efficient
*and* fast.

## 4. Why QLoRA is useful on limited GPU

Because the frozen base weights stay in 4-bit form, QLoRA lets you fine-tune models that
would otherwise not fit on a single GPU (e.g. fitting a 7B-parameter model on a free
Colab T4 with ~15GB VRAM). Combined with LoRA's tiny trainable parameter count and
techniques like gradient checkpointing, QLoRA makes it possible to fine-tune meaningfully
sized models on free or low-cost hardware, which is exactly the setup used in this
project (Colab T4, 4-bit Qwen2.5-0.5B).

## 5. What is non-instruction fine-tuning?

Non-instruction fine-tuning (also called domain-adaptive pretraining) continues
next-token-prediction training on **raw, unstructured domain text** — no
instruction/response structure, just plain paragraphs (in this project, cleaned raw
e-commerce support text). The goal is purely to shift the model's internal language
distribution toward domain vocabulary and recurring facts (e.g. "warehouse packing
status", "wallet credit", "firmware update") before teaching it how to answer questions.

## 6. What is instruction fine-tuning?

Instruction fine-tuning (SFT — Supervised Fine-Tuning) trains the model on paired
**instruction → response** examples, formatted with a consistent prompt template
(`### Instruction: ... ### Response: ...`). This teaches the model the *behavior* of
answering a user's question helpfully and correctly, rather than just continuing text.
In this project, this stage uses `data/instruction_dataset.jsonl` (100 Q&A pairs covering
ten e-commerce support topics).

## 7. What is DPO?

DPO (Direct Preference Optimization) is a way to align a model with human/curated
preferences **without** needing a separate reward model (unlike classic RLHF/PPO).
Given a prompt with a **chosen** (preferred) and a **rejected** (dispreferred) response,
DPO directly optimizes the model's parameters so the probability of generating the
chosen response increases relative to the rejected one, using a contrastive loss
controlled by a temperature parameter `beta`. ORPO is a related, slightly newer
technique that combines preference optimization with the SFT loss in a single training
objective, removing the need for a separate SFT stage before alignment — but in this
project we use the standard two-step SFT-then-DPO pipeline as required by the
assignment.

## 8. Difference between SFT and DPO

| | SFT | DPO |
|---|---|---|
| Data | Single "correct" response per prompt | Paired chosen vs rejected responses per prompt |
| Learns | How to produce *a* valid answer | Which of two answers is *better* |
| Loss | Standard next-token cross-entropy | Contrastive preference loss (log-ratio of chosen vs rejected likelihoods) |
| Effect | Adds knowledge/behavior the model didn't have | Refines tone, safety, and quality of behavior the model already has |

In short: SFT teaches the model *what* to know and *how* to format an answer; DPO
teaches it *which style* of correct-ish answer is actually preferred.

## 9. Hyperparameter values used in this project

| Parameter | Stage 1 (Non-Instruction) | Stage 2 (Instruction SFT) | Stage 3 (DPO) |
|---|---|---|---|
| Rank (`r`) | 16 | 16 | 16 |
| Alpha (`lora_alpha`) | 16 | 16 | 16 |
| Dropout (`lora_dropout`) | 0.0 | 0.0 | 0.0 |
| Learning rate | 2e-4 | 1e-4 | 5e-6 |
| Batch size (per device) | 2 (grad accum 4) | 2 (grad accum 4) | 2 (grad accum 4) |
| Max steps | 60 | 90 | 60 |
| Beta (DPO only) | — | — | 0.1 |

**Why these values:** rank 16 / alpha 16 is a common, balanced default for small models
that gives the adapter enough capacity without overfitting on a small dataset. The
learning rate is highest for Stage 1 (broad domain-language adaptation), lower for
Stage 2 (more precise instruction-following behavior), and much lower for Stage 3 (DPO
is sensitive to large updates since it's refining an already-trained model's
preferences, not teaching it new facts). Batch size of 2 with gradient accumulation of 4
(effective batch size 8) fits comfortably within a free-tier Colab T4 GPU's VRAM for a
0.5B model in 4-bit.
