# Counter Speech Generation With Facts

_A robust framework for generating fact-based, context-aware counterspeech using large language models and retrieval-augmented generation._

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methods](#methods)
- [Model Details](#model-details)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Fine-Tuning & Experimentation](#fine-tuning--experimentation)
- [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)

---

## Overview

The proliferation of hate speech on digital platforms poses significant risks to individuals and society. While automated counterspeech generation has emerged as a promising strategy, most existing systems lack factual grounding and persuasive impact.

**This project proposes a novel framework that combines large pre-trained language models (BLOOMZ, FLAN-T5-XL) with external knowledge retrieval to generate context-aware, non-aggressive, and fact-based counterspeech.**  
We leverage advanced fine-tuning techniques (LoRA, instruction tuning, prefix/prompt tuning) and retrieval-augmented generation (RAG) to ensure responses are both relevant and verifiable.

---

## Dataset

- **Source:** [DIALOCONAN](https://aclanthology.org/2022.emnlp-main.549.pdf)
- **Targets:** JEWS, LGBT+, MIGRANTS, MUSLIMS, PEOPLE OF COLOR (POC), WOMEN, and rare intersectional cases.
- **Structure:** 3,059 multi-turn dialogues (16,625 turns), each turn annotated with text, target, dialogue_id, turn_id, type (HS or CN), and source.
- **Splitting:** No predefined splits; typical practice is 80% train, 10% validation, 10% test.

| Target      | Dialogues | Percentage |
|-------------|-----------|------------|
| LGBT+       | 591       | 19.32%     |
| MIGRANTS    | 534       | 17.46%     |
| MUSLIMS     | 505       | 16.51%     |
| POC         | 493       | 16.12%     |
| JEWS        | 468       | 15.30%     |
| WOMEN       | 462       | 15.10%     |
| Other       | 6         | 0.20%      |
| **Total**   | **3059**  | **100%**   |

---

## Methods

- **LoRA (Low-Rank Adaptation):** Efficient fine-tuning by inserting small trainable matrices into transformer layers, keeping base weights frozen.
- **PEFT:** Parameter-efficient fine-tuning library supporting LoRA, prefix tuning, and prompt tuning.
- **Instruction Tuning:** Fine-tuning on instruction-response pairs for generalization to novel prompts.
- **Prefix/Prompt Tuning:** Training small sets of virtual tokens or embeddings to steer model behavior without updating core weights.

---

## Model Details

- **BLOOMZ:** Multilingual, instruction-tuned model, strong for English and zero-shot/few-shot tasks.
- **FLAN-T5-XL:** Instruction-tuned T5 model, excels at text-to-text tasks and robust to diverse prompts.
- **Hyperparameters:** LoRA rank (r), alpha scaling, and target modules are optimized for each model.

---

## Data Preprocessing Pipeline

1. **Load and prepare DIALOCONAN dataset.**
2. **Group turns by dialogue_id** and sort by turn_id.
3. **Construct dialogue histories** for each counterspeech (CN) turn, forming (context, target) pairs.
4. **Tokenize** inputs and targets for model training.

---

## Fine-Tuning & Experimentation

- Models are fine-tuned using LoRA, instruction tuning, prefix tuning, and prompt tuning.
- Hyperparameters (LoRA rank, alpha, target modules) are systematically varied for optimal results.
- Evaluation metrics: **BERTScore, ROUGE, BLEU, Perplexity, Toxicity**.

---

## Retrieval-Augmented Generation (RAG)

To enhance factuality, we implement a lightweight RAG pipeline:

1. **Retriever** (SentenceTransformer + FAISS) encodes the user dialogue and retrieves top-k relevant factual statements from a knowledge base.
2. **Facts are prepended** to the dialogue history to form a rich prompt.
3. **LoRA-tuned FLAN-T5** generates a factual, counterspeech response.

**Prompt Example:**
