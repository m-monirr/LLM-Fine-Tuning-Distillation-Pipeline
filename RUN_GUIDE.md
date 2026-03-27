# 🚀 Complete Run Guide - Arabic News Assistant

This guide provides detailed, step-by-step instructions to run the entire finetuning pipeline from scratch.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [API Keys Configuration](#3-api-keys-configuration)
4. [Data Preparation](#4-data-preparation)
5. [Knowledge Distillation](#5-knowledge-distillation)
6. [Model Training](#6-model-training)
7. [Model Evaluation](#7-model-evaluation)
8. [Inference & Deployment](#8-inference--deployment)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

### Hardware Requirements

#### Minimum
- **CPU**: 4 cores
- **RAM**: 16GB
- **Storage**: 50GB free space
- **GPU**: None (CPU training possible but slow)

#### Recommended
- **CPU**: 8+ cores
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 3070, etc.)

#### Optimal
- **CPU**: 16+ cores
- **RAM**: 64GB
- **Storage**: 200GB NVMe SSD
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100)

### Software Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python**: 3.8, 3.9, 3.10, or 3.11 (recommended: **3.11**)
- **CUDA**: 11.8 or 12.1 (if using GPU)
- **Git**: Latest version

> Tip (Windows): use Python **3.11** for best compatibility. Python 3.13 may fail to install some pinned dependencies (e.g., `numpy<2.0.0`) without Visual Studio build tools.

### Verify Your System

```bash
# Check Python version
python --version  # Should show 3.8+

# Check CUDA (if GPU)
nvidia-smi  # Should show your GPU

# Check available memory
# Windows
wmic OS get FreePhysicalMemory
# Linux/Mac
free -h
```

---

## 2. Environment Setup

### Step 2.1: Clone the Repository

```bash
# Clone the project
git clone <repository-url>
cd arabic-news-assistant

# Or if you're starting from scratch
mkdir arabic-news-assistant
cd arabic-news-assistant
```

### Step 2.2: Create Virtual Environment

#### On Windows
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (should show (venv) in prompt)
```

#### On Linux/Mac
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show (venv) in prompt)
```

### Step 2.3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only version
pip install torch torchvision torchaudio

# Install project dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
PyTorch: 2.1.0+cu118
CUDA Available: True  # or False for CPU
```

---

## 3. API Keys Configuration

### Step 3.1: Get API Keys

You need three API keys:

#### Hugging Face Token
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name: "arabic-news-assistant"
4. Type: "Write"
5. Copy the token

#### OpenRouter API Key
1. Go to [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Click "Create API Key"
3. Name: "arabic-news-distillation"
4. Copy the key

#### Weights & Biases Key
1. Go to [https://wandb.ai/authorize](https://wandb.ai/authorize)
2. Copy your API key

### Step 3.2: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file
# Windows: notepad .env
# Linux/Mac: nano .env
```

Update the `.env` file:
```env
HUGGINGFACE_TOKEN=hf_your_actual_token_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
WANDB_API_KEY=your_actual_wandb_key_here
```

### Step 3.3: Verify API Keys

```python
# Run this in Python interpreter or notebook
from dotenv import load_dotenv
import os

load_dotenv()

print("HF Token:", os.getenv("HUGGINGFACE_TOKEN")[:10] + "...")
print("OpenRouter:", os.getenv("OPENROUTER_API_KEY")[:10] + "...")
print("W&B:", os.getenv("WANDB_API_KEY")[:10] + "...")
```

---

## 4. Data Preparation

### Step 4.1: Prepare Your Data Directory

```bash
# Create data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p outputs/models
mkdir -p outputs/logs
```

### Step 4.2: Add Your News Data

For a quick demo, this repo already includes a small sample dataset at `datasets/news-sample.jsonl`.

To use your own dataset, place Arabic news articles in `data/raw/news.jsonl` (or any path you prefer) and update the paths in `config/training_config.yaml` and your scripts accordingly.

**Required format:**
```json
{"content": "نص الخبر العربي هنا..."}
{"content": "خبر آخر..."}
```

### Step 4.3: Validate Data Format

```python
import json

# Validate data
with open("datasets/news-sample.jsonl", "r", encoding="utf-8") as f:
    count = 0
    for line in f:
        if line.strip():
            data = json.loads(line)
            assert "content" in data, "Missing 'content' field"
            count += 1
    print(f"✅ Validated {count} news articles")
```

---

## 5. Knowledge Distillation

This step generates training data using DeepSeek API.

### Step 5.1: Run Data Generation

#### Option A: Use Notebook (Recommended for Beginners)

```bash
jupyter notebook notebooks/main_pipeline.ipynb
```

Navigate to the "Data Generation" section and run the cells.

#### Option B: Use Python Script

```python
import os
from dotenv import load_dotenv

from src.data_generator import DataGenerator
from src.data_loader import DataLoader

load_dotenv()

# Load raw data
raw_data = DataLoader.load_raw_data("datasets/news-sample.jsonl")
print(f"Loaded {len(raw_data)} articles")

# Initialize generator
generator = DataGenerator(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_id="deepseek/deepseek-v3.1-terminus",
    temperature=0.2,
    max_tokens=512
)

# Generate SFT dataset
generator.generate_sft_dataset(
    raw_data=raw_data[:50],  # Start with 50 samples
    output_path="data/processed/sft.jsonl",
    include_translations=True,
    target_languages=["English", "French"],
    max_samples=50
)
```

### Step 5.2: Monitor Progress

The generation process will show:
```
Processing articles: 100%|████████| 50/50 [05:23<00:00, 6.47s/it]
✅ Generated 50 extraction examples
✅ Generated 100 translation examples
✅ Total: 150 training examples
✅ Success rate: (varies by model/prompt/data)
📁 Saved to: data/processed/sft.jsonl
```

### Step 5.3: Verify Generated Data

```python
# Check generated data
with open("data/processed/sft.jsonl", "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]
    
print(f"Total samples: {len(samples)}")
print(f"Successful: {sum(1 for s in samples if s['status'] == 'success')}")
print(f"Failed: {sum(1 for s in samples if s['status'] == 'failed')}")

# Show example
print("\nSample:")
print(json.dumps(samples[0], indent=2, ensure_ascii=False)[:500])
```

---

## 6. Model Training

### Step 6.1: Configure Training

Edit `config/training_config.yaml` if needed:

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4  # Reduce if GPU memory is low
  gradient_accumulation_steps: 4   # Increase if batch size is reduced
  learning_rate: 2.0e-4
```

### Step 6.2: Start Training

#### Option A: Using Notebook (Recommended)

```bash
jupyter notebook notebooks/main_pipeline.ipynb
```

Run the "Training" section cells.

#### Option B: Using Python Script

```python
import os
from dotenv import load_dotenv

from src.trainer import ModelTrainer
from src.data_loader import DataLoader

load_dotenv()

# Load and prepare dataset
print("Loading dataset...")
train_dataset, eval_dataset = DataLoader.load_and_prepare(
    sft_data_path="data/processed/sft.jsonl",
    test_size=0.1,
    seed=42
)

# Initialize trainer
print("Initializing trainer...")
trainer = ModelTrainer(config_path="config/training_config.yaml")

# Load model
print("Loading base model...")
trainer.load_model(hf_token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare trainer
print("Preparing trainer...")
trainer.prepare_trainer(train_dataset, eval_dataset)

# Start training
print("Starting training...")
trainer.train()

# Save model
print("Saving model...")
trainer.save_model("outputs/models/qwen-arabic-assistant-final")
```

### Step 6.3: Monitor Training

Training will show progress:
```
Epoch 1/3:  33%|███▎      | 100/300 [02:15<04:30, 1.35s/it]
Loss: 0.4523 | Grad Norm: 0.832 | LR: 0.0002

{'loss': 0.4523, 'learning_rate': 0.0002, 'epoch': 1.0}
{'eval_loss': 0.5234, 'eval_runtime': 15.23, 'epoch': 1.0}
```

**Training Time Estimates:**
- RTX 3060 (12GB): ~3-4 hours
- RTX 3090 (24GB): ~2 hours
- A100 (40GB): ~1 hour
- CPU: ~24-48 hours (not recommended)

### Step 6.4: Track with Weights & Biases

1. Open browser: [https://wandb.ai](https://wandb.ai)
2. Navigate to your project
3. Monitor:
   - Training loss curve
   - Validation loss
   - GPU utilization
   - Learning rate schedule

---

## 7. Model Evaluation

### Step 7.1: Basic Inference Test

```python
from src.inference import ModelInference

# Load model
inference = ModelInference(
    model_path="outputs/models/qwen-arabic-assistant-final"
)

# Test extraction
test_story = """
أعلن الرئيس المصري عبد الفتاح السيسي عن إطلاق مبادرة جديدة
لتطوير قطاع التعليم بتكلفة تصل إلى 50 مليار جنيه مصري.
"""

result = inference.extract_details(test_story)
print("Extraction Result:")
print(result)
```

### Step 7.2: Evaluate Translation

```python
# Test translation
translation = inference.translate(test_story, "English")
print("\nTranslation Result:")
print(translation)
```

### Step 7.3: Batch Evaluation

```python
# Evaluate on test set
from src.data_loader import DataLoader
import json

sft_data = DataLoader.load_sft_data("data/processed/sft.jsonl")
test_samples = sft_data[:10]  # Use first 10 for quick test

results = []
for sample in test_samples:
    prediction = inference.extract_details(sample['story'])
    results.append({
        'story': sample['story'][:100],
        'expected': sample['response'],
        'predicted': prediction
    })

# Save results
with open("outputs/evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Evaluation complete. Check outputs/evaluation_results.json")
```

---

## 8. Inference & Deployment

### Step 8.1: Interactive Testing

Create a simple test script `test_model.py`:

```python
from src.inference import ModelInference
import json

def main():
    # Load model
    print("Loading model...")
    inference = ModelInference(
        model_path="outputs/models/qwen-arabic-assistant-final"
    )
    print("✅ Model loaded successfully!\n")
    
    while True:
        print("\n" + "="*50)
        print("Arabic News Assistant - Interactive Test")
        print("="*50)
        print("\nOptions:")
        print("1. Extract details")
        print("2. Translate to English")
        print("3. Translate to French")
        print("4. Quit")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == "4":
            break
        
        story = input("\nEnter Arabic news story:\n")
        
        if choice == "1":
            result = inference.extract_details(story)
        elif choice == "2":
            result = inference.translate(story, "English")
        elif choice == "3":
            result = inference.translate(story, "French")
        else:
            print("Invalid choice")
            continue
        
        print("\n" + "="*50)
        print("Result:")
        print("="*50)
        print(result)

if __name__ == "__main__":
    main()
```

Run it:
```bash
python test_model.py
```

### Step 8.2: Create API Endpoint (Optional)

Install FastAPI:
```bash
pip install fastapi uvicorn
```

Create `api.py`:
```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import ModelInference

app = FastAPI()
inference = ModelInference("outputs/models/qwen-arabic-assistant-final")

class StoryRequest(BaseModel):
    story: str
    task: str = "extract"
    target_language: str = "English"

@app.post("/process")
def process_story(request: StoryRequest):
    if request.task == "extract":
        result = inference.extract_details(request.story)
    elif request.task == "translate":
        result = inference.translate(request.story, request.target_language)
    else:
        result = inference.generate(request.story, request.task)
    
    return {"result": result}

# Run with: uvicorn api:app --reload
```

---

## 9. Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in `config/training_config.yaml`:
   ```yaml
   per_device_train_batch_size: 2  # or 1
   ```

2. Increase gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 8  # or 16
   ```

3. Reduce max sequence length:
   ```yaml
   max_seq_length: 1024  # instead of 2048
   ```

### Issue 2: NumPy Version Conflict

**Symptoms:**
```
ValueError: numpy.dtype size changed
```

**Solution:**
```bash
pip uninstall numpy -y
pip install numpy==1.24.3
```

### Issue 3: Slow Training

**Symptoms:** Training taking too long

**Solutions:**
1. Enable gradient checkpointing (already enabled)
2. Use mixed precision training (already enabled)
3. Reduce dataset size for testing:
   ```python
   train_dataset = train_dataset.select(range(100))
   ```

### Issue 4: API Rate Limits

**Symptoms:**
```
Error 429: Too Many Requests
```

**Solution:**
Add rate limiting in `src/data_generator.py`:
```python
import time

# Add sleep between requests
time.sleep(2)  # 2 seconds between API calls
```

### Issue 5: Model Not Loading

**Symptoms:**
```
OSError: Model not found
```

**Solution:**
```python
# Verify model path
import os
model_path = "outputs/models/qwen-arabic-assistant-final"
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Files: {os.listdir(model_path)}")
```

---

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check Issues**: [GitHub Issues](https://github.com/yourusername/arabic-news-assistant/issues)
2. **Ask Community**: [GitHub Discussions](https://github.com/yourusername/arabic-news-assistant/discussions)
3. **Read Logs**: Check `outputs/logs/` for detailed error messages
4. **Enable Debug Mode**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

---

## ✅ Success Checklist

- [ ] Environment setup complete
- [ ] API keys configured
- [ ] Raw data prepared
- [ ] SFT dataset generated
- [ ] Model training completed
- [ ] Model saved successfully
- [ ] Inference working
- [ ] Results satisfactory

---

**Happy Training! 🚀**
