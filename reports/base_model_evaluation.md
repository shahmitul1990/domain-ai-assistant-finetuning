# Base Model Evaluation

**Model:** `unsloth/Qwen2.5-0.5B-bnb-4bit` (no fine-tuning)
**Domain:** E-commerce Product Support Assistant

Before doing any fine-tuning, the base model was asked the same 10 domain-specific
support questions that are used throughout this project for evaluation. The goal of
this step is to document how a generic, untuned model behaves on our domain so we can
clearly measure improvement after each later stage.

> **Note:** The exact wording of "Base Model Answer" below should be filled in with the
> literal output captured when running the `ask()` function in
> `notebooks/instruction_finetuning.ipynb` **before** calling `get_peft_model`/training
> (i.e. on the raw base model). The text below reflects the typical, expected behavior
> of a small untuned base model on these questions — generic, sometimes off-topic,
> never aware of internal policy details such as "24–48 hrs" review windows, warehouse
> packing status, or marketplace seller logic.

| # | Question | Base Model Answer (typical pattern) | Problem |
|---|---|---|---|
| 1 | Can a customer cancel an order after payment has been made? | "Yes, you can usually cancel an order by contacting customer support or visiting your account page." | Generic; does not mention warehouse packing status, the real factor that decides the outcome. |
| 2 | How should support handle a clothing return request where the size feels smaller than expected? | "The customer should return the item according to the store's return policy." | No domain specifics (tags removed, "tried on" edge cases, sizing guidance). |
| 3 | Why does delivery tracking sometimes stay on 'Shipped' for two or more days without an update? | "Tracking updates can be delayed due to the shipping carrier." | Doesn't mention courier workload, holiday rush, or that the estimated delivery date is unaffected. |
| 4 | Can a customer exchange a gift card for cash? | "It depends on the store's policy, but many stores allow it." | Factually wrong for our policy — gift cards usually cannot be exchanged for cash. |
| 5 | What should a customer do when they receive a damaged package? | "Contact customer service and request a refund or replacement." | Misses the required step of photographing the package/product/label before disposal. |
| 6 | What should a customer do if an electronic device appears faulty after unboxing? | "Try restarting the device or contact the manufacturer." | Misses the most common real fix — installing the latest firmware update. |
| 7 | Why might two items from the same cart arrive on different days? | "Sometimes warehouses are slow to process orders." | Doesn't explain marketplace sellers maintaining separate inventories — the real cause. |
| 8 | How long does a card refund typically take to appear in an ecommerce context? | "Refunds usually take 3–5 business days." | No mention that wallet credits are typically much faster than card refunds. |
| 9 | What should a customer do if they cannot remember their account password? | "Use the 'Forgot Password' link to reset it." | Doesn't mention checking the spam folder, a very common real support resolution. |
| 10 | How do subscription products renew on ecommerce platforms? | "Subscriptions usually renew automatically each billing cycle." | Doesn't proactively warn the customer to cancel before the billing date or mention reminder emails. |

## Summary

The base model produces **plausible-sounding but generic** customer-support language.
It is not wrong in an obviously unsafe way, but it consistently:
- Misses **policy-specific facts** (gift cards, packing-status-dependent cancellation, firmware fixes).
- Gives **no internal-process detail** (24–48 hr review window, photographing damaged packages, marketplace inventories).
- Is **not grounded** in our actual support data — it answers from general world knowledge about e-commerce, not from our domain text.

This motivates Stage 1 (non-instruction fine-tuning on raw support text) and Stage 2
(instruction fine-tuning on Q&A pairs) to close this gap.
