"""
Model inference utilities.
"""

import torch
import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel

# Load environment variables
load_dotenv()


class ModelInference:
    """Handle model inference."""
    
    def __init__(self, model_path: str, max_seq_length: int = 2048, hf_token: str = None):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Determine precision based on GPU capabilities
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
        
        self.model, self.tokenizer = self._load_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load finetuned model."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=torch.cuda.is_available(),
            token=self.hf_token
        )
        return model, tokenizer
    
    def generate(
        self,
        story: str,
        task: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate response for given story and task."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
            {"role": "user", "content": f"{story}\n\nTask: {task}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "ASSISTANT:" in response:
            assistant_response = response.split("ASSISTANT:")[-1].strip()
        else:
            assistant_response = response
        
        return assistant_response
    
    def extract_details(self, story: str) -> str:
        """Extract structured details from story."""
        return self.generate(
            story=story,
            task="Extract the story details into a JSON.",
            temperature=0.3
        )
    
    def translate(self, story: str, target_language: str) -> str:
        """Translate story to target language."""
        return self.generate(
            story=story,
            task=f"Translate the story into {target_language} and return JSON.",
            temperature=0.5
        )
