# Arabic News Assistant - Qwen Finetuning Project

A complete pipeline for finetuning Qwen models on Arabic news data for tasks like information extraction, translation, and entity recognition.

## Project Structure
````markdown
# Arabic News Assistant Finetuning Project

This project uses Unsloth to finetune a Qwen2 model on Arabic news data for tasks like information extraction, translation, and summarization.

## Setup

1. Install dependencies:
   ```bash
   pip install -qU transformers datasets optimum
   pip install -qU openai wandb
   pip install -qU json-repair faker vllm
   pip install -qU unsloth bitsandbytes accelerate peft trl
   ```

2. API Keys:
   - Create `api_credentials.py` with your API keys
   - Required keys: Hugging Face, OpenRouter, Weights & Biases

3. Data:
   - Place your news-sample.jsonl file in the `data` folder
   - If you have existing SFT data, place it in `data/sft.jsonl`

## Notebooks

- `unslouth_local.ipynb`: Local version for running on your machine
- `unslouth.ipynb`: Google Colab version for cloud-based training

## Model Architecture

This project uses:
- Base model: Qwen2-1.5B-Instruct
- Finetuning method: LoRA (Low-Rank Adaptation)
- LoRA settings: r=16, alpha=16
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Training Parameters

- Epochs: 3
- Batch size: Automatically adjusted based on GPU memory
- Precision: BF16 (for Ampere+ GPUs) or FP16
- Learning rate: 2e-4
- Sequence length: 2048 tokens

## Usage

After training, the model can be used for:
- Extracting structured data from Arabic news articles
- Translating articles to other languages
- Identifying entities and categorizing news content
````
