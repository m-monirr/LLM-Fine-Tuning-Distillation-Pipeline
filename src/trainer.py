"""
Model training with Unsloth optimization.
"""

import torch
import os
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Load environment variables
load_dotenv()


class ModelTrainer:
    """Handle model training with Unsloth."""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[dict] = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model(self, hf_token: Optional[str] = None):
        """Load and prepare model with LoRA."""
        model_config = self.config['model']
        lora_config = self.config['lora']
        
        # Use token from parameter, environment, or None
        token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Determine dtype
        if model_config['dtype'] == 'bfloat16':
            dtype = torch.bfloat16
        elif model_config['dtype'] == 'float16':
            dtype = torch.float16
        else:
            dtype = None
        
        # Load model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['base_model_id'],
            max_seq_length=model_config['max_seq_length'],
            dtype=dtype,
            load_in_4bit=model_config['load_in_4bit'],
            token=token
        )
        
        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config['r'],
            target_modules=lora_config['target_modules'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            use_gradient_checkpointing=True,
            random_state=self.config['training']['seed']
        )
        
        print("✅ Model loaded and LoRA configured")
        return self.model, self.tokenizer
    
    def prepare_trainer(self, train_dataset, eval_dataset):
        """Prepare SFT trainer."""
        training_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            gradient_checkpointing=training_config['gradient_checkpointing'],
            optim=training_config['optim'],
            logging_steps=training_config['logging_steps'],
            save_strategy=training_config['save_strategy'],
            save_steps=training_config['save_steps'],
            evaluation_strategy=training_config.get('evaluation_strategy', 'steps'),
            eval_steps=training_config.get('eval_steps', 100),
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            fp16=training_config['fp16'],
            bf16=training_config['bf16'],
            max_grad_norm=training_config['max_grad_norm'],
            warmup_ratio=training_config['warmup_ratio'],
            group_by_length=training_config['group_by_length'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            report_to=training_config['report_to'],
            seed=training_config['seed']
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="messages",
            args=training_args,
            packing=True,
            max_seq_length=self.config['model']['max_seq_length']
        )
        
        print("✅ Trainer configured")
        return self.trainer
    
    def train(self):
        """Start training."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call prepare_trainer() first.")
        
        print("🚀 Starting training...")
        self.trainer.train()
        print("✅ Training complete!")
    
    def save_model(self, output_path: str):
        """Save trained model."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        print(f"✅ Model saved to {output_path}")
