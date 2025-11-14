"""
Load and prepare datasets for finetuning.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from datasets import Dataset


class DataLoader:
    """Load and prepare datasets for training."""
    
    @staticmethod
    def load_raw_data(file_path: str) -> list:
        """Load raw JSONL data."""
        raw_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
        return raw_data
    
    @staticmethod
    def load_sft_data(file_path: str, filter_successful: bool = True) -> list:
        """Load SFT data from JSONL file."""
        sft_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sft_data.append(json.loads(line))
        
        if filter_successful:
            sft_data = [item for item in sft_data if item.get("status") == "success"]
        
        return sft_data
    
    @staticmethod
    def prepare_chat_dataset(
        sft_data: list,
        system_message: str = "You are a helpful assistant that follows instructions precisely."
    ) -> list:
        """Convert SFT data to chat format."""
        chat_data = []
        
        for item in sft_data:
            # Ensure response is always a string
            if isinstance(item["response"], dict):
                assistant_response = json.dumps(item["response"], ensure_ascii=False)
            else:
                assistant_response = str(item["response"])
            
            chat_example = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"{item.get('story', '')}\n\nTask: {item.get('task', '')}"},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            chat_data.append(chat_example)
        
        return chat_data
    
    @staticmethod
    def create_train_eval_split(
        chat_data: list,
        test_size: float = 0.1,
        seed: int = 42
    ) -> Tuple[Dataset, Dataset]:
        """Create train/eval datasets."""
        dataset_df = pd.DataFrame(chat_data)
        hf_dataset = Dataset.from_pandas(dataset_df)
        
        train_test_split = hf_dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        return train_dataset, eval_dataset
    
    @classmethod
    def load_and_prepare(
        cls,
        sft_data_path: str,
        test_size: float = 0.1,
        seed: int = 42
    ) -> Tuple[Dataset, Dataset]:
        """Complete pipeline: load SFT data and prepare train/eval splits."""
        sft_data = cls.load_sft_data(sft_data_path, filter_successful=True)
        print(f"Loaded {len(sft_data)} successful examples")
        
        chat_data = cls.prepare_chat_dataset(sft_data)
        train_dataset, eval_dataset = cls.create_train_eval_split(
            chat_data, test_size=test_size, seed=seed
        )
        
        print(f"Training examples: {len(train_dataset)}, Evaluation examples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
