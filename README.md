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
Facts: [fact1] [fact2] [fact3]
Conversation: [dialogue turn 1] [SEP] [dialogue turn 2] ...
Response:

---

## Results

| Model                | BERTScore F1 | ROUGE-1 F1 | ROUGE-L F1 | Perplexity | Toxicity |
|----------------------|--------------|------------|------------|------------|----------|
| BLOOMZ (LoRA)        | 0.62         | 0.04       | 0.04       | 1.00       | 0.002    |
| FLAN-T5-XL (LoRA)    | 0.70         | 0.12       | 0.09       | 1.00       | 0.08     |
| FLAN-T5-XL (RAG+LoRA)| 0.73         | 0.14       | 0.11       | 1.00       | 0.07     |

---

2. **Prepare the dataset:**  
Download DIALOCONAN or use the Hugging Face datasets loader.
3. **Train or load a LoRA-tuned model:**  
See `train_lora.py` or similar scripts.
4. **Run RAG inference:**  
Use `generate_with_rag.py` to generate fact-based counterspeech.
5. **Evaluate:**  
Use `evaluate.py` for metrics.

---

## References

- Tekiroğlu, S., Saha, P., & Guerini, M. (2022). DIALOCONAN: A Dialogue Dataset for Counterspeech Generation. [EMNLP 2022 PDF](https://aclanthology.org/2022.emnlp-main.549.pdf)
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Chung, H. W., et al. (2022). Scaling Instruction-Finetuned Language Models. [arXiv:2210.11416](https://arxiv.org/abs/2210.11416)
- Hugging Face PEFT Documentation: https://huggingface.co/docs/peft

---

**For full details, see [Counter Speech Generation.pdf](NLP_Project_Report(1).pdf) included in this repository.**

---

