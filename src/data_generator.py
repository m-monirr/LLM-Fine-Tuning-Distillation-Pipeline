"""
Generate SFT data using knowledge distillation from larger models via OpenRouter API.
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path
import requests
from tqdm.auto import tqdm
import json_repair
from dotenv import load_dotenv

from .models import NewsDetails, TranslatedStory

# Load environment variables
load_dotenv()


class DataGenerator:
    """Generate training data using OpenRouter API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_id: str = "deepseek/deepseek-v3.1-terminus",
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: int = 60
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY in .env file or pass it directly.")
        
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
    def _parse_json(self, text: str) -> Optional[dict]:
        """Parse JSON from text response with error handling."""
        try:
            return json_repair.loads(text)
        except:
            return None
    
    def _fix_json(self, text: str) -> Optional[dict]:
        """Attempt to clean and parse JSON from text response."""
        try:
            text = text.strip()
            if text.startswith("{") and text.endswith("}"):
                return json.loads(text)
            else:
                return self._parse_json(text)
        except:
            return None

    def _call_openrouter_api(self, prompt: str) -> str:
        """Call OpenRouter API with the given prompt."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_id,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        response = requests.post(
            "https://api.openrouter.ai/v1/engines/deepseek/completions",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
    
    def generate_data(self, prompts: List[str]) -> List[dict]:
        """Generate data for the given prompts."""
        results = []
        for prompt in tqdm(prompts, desc="Generating data"):
            try:
                text = self._call_openrouter_api(prompt)
                json_data = self._fix_json(text)
                if json_data:
                    results.append(json_data)
                else:
                    print(f"Warning: Failed to parse JSON from response: {text}")
            except Exception as e:
                print(f"Error processing prompt '{prompt}': {e}")
        return results

    def generate_news_details(self, headlines: List[str]) -> List[NewsDetails]:
        """Generate news details for the given headlines."""
        prompts = [f"Write a detailed news article about: {headline}" for headline in headlines]
        data = self.generate_data(prompts)
        return [NewsDetails(**item) for item in data if item]

    def generate_translated_stories(self, stories: List[str], target_language: str) -> List[TranslatedStory]:
        """Generate translated stories for the given stories."""
        prompts = [f"Translate this story to {target_language}: {story}" for story in stories]
        data = self.generate_data(prompts)
        return [TranslatedStory(**item) for item in data if item]