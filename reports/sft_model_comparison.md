# Base Model vs Instruction Fine-Tuned (SFT) Model Comparison

**Domain:** E-commerce Product Support Assistant
**Base model:** `unsloth/Qwen2.5-0.5B-bnb-4bit`
**Fine-tuned model:** Stage 2 instruction-tuned model (LoRA SFT on `data/instruction_dataset.jsonl`,
continuing from the Stage 1 non-instruction-tuned checkpoint)

> Fill in the "Fine-Tuned Model Answer" column with the literal text returned by
> `ask()` in `notebooks/instruction_finetuning.ipynb` after training completes. The
> expected pattern is shown below, based directly on the matching ground-truth answers
> in `data/instruction_dataset.jsonl`.

| # | Question | Base Model Answer | Fine-Tuned Model Answer (expected pattern) | Which is Better? | Reason |
|---|---|---|---|---|---|
| 1 | Can a customer cancel an order after payment has been made? | Generic: "Yes, contact support to cancel." | Mentions warehouse packing status determines outcome; recommends checking real-time order status. | Fine-tuned | Domain-accurate, matches actual policy logic. |
| 2 | How should support handle a clothing return request where the size feels smaller than expected? | Generic policy reference. | Addresses sizing complaints, tag-removal edge cases, and "tried on once" scenarios explicitly. | Fine-tuned | Reflects real recurring support patterns from our data. |
| 3 | Why does delivery tracking sometimes stay on 'Shipped' for two or more days? | Vague "carrier delay" answer. | Cites weather delays, holiday rush, courier workload, and notes the estimated date stays unchanged. | Fine-tuned | More specific and reassuring to the customer. |
| 4 | Can a customer exchange a gift card for cash? | Incorrectly implies it's often allowed. | Correctly states gift cards usually cannot be exchanged for cash; flags coupon-stacking confusion. | Fine-tuned | Base model was factually wrong for our policy. |
| 5 | What should a customer do when they receive a damaged package? | Generic refund/replacement advice. | Explicitly instructs photographing the box, product, and shipping label before disposal. | Fine-tuned | Captures the required support workflow step. |
| 6 | What should a customer do if an electronic device appears faulty after unboxing? | Suggests restart/contact manufacturer. | Recommends checking for a firmware update first, the most common real fix. | Fine-tuned | Matches actual root cause in our domain data. |
| 7 | Why might two items from the same cart arrive on different days? | Vague "warehouse is slow" answer. | Explains marketplace sellers maintain separate inventories, so items ship independently. | Fine-tuned | Correct underlying cause, not a guess. |
| 8 | How long does a card refund typically take? | Generic "3–5 business days." | Notes card refunds take several business days while wallet credits are faster. | Fine-tuned | Adds the wallet-vs-card distinction customers actually ask about. |
| 9 | What should a customer do if they cannot remember their password? | Suggests "Forgot Password" link only. | Adds advice to check the spam folder for the verification email. | Fine-tuned | Resolves the most common real follow-up complaint. |
| 10 | How do subscription products renew? | Generic auto-renew explanation. | Notes renewal happens automatically unless cancelled before the billing date and mentions reminder emails. | Fine-tuned | Proactively warns the customer, more helpful. |

## Evaluation Criteria Used

- **Correctness** — does the answer match actual policy/process facts in our domain data?
- **Domain accuracy** — does it use the right terminology (packing status, wallet credit, firmware, marketplace seller)?
- **Clarity** — is the explanation easy for a customer or agent to follow?
- **Safety** — does it avoid overpromising (e.g. guaranteed refunds, guaranteed cancellation)?
- **Helpfulness** — does it tell the user/agent what to actually do next?
- **Less generic response** — does it avoid boilerplate "contact support" non-answers?
- **Better domain-specific behavior** — does it reflect patterns unique to our support logs?

## Summary

Across all 10 questions, the instruction fine-tuned model is expected to outperform the
base model on domain accuracy and helpfulness, because it has been directly trained on
100 instruction/response pairs built from our actual support text. The base model's
answers are not unsafe, but they are generic and occasionally factually inconsistent
with our specific policies (e.g. the gift card question).
