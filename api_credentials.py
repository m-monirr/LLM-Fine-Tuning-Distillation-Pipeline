"""
Credentials file for API tokens.
This file loads API tokens from environment variables.
IMPORTANT: Configure your tokens in the .env file!
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Tokens from environment
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

def get_tokens():
    """Returns a dictionary of all API tokens."""
    return {
        "huggingface": HUGGINGFACE_TOKEN,
        "openrouter": OPENROUTER_API_KEY,
        "wandb": WANDB_API_KEY
    }

def get_hf_token():
    """Returns the Hugging Face token."""
    return HUGGINGFACE_TOKEN

def get_openrouter_token():
    """Returns the OpenRouter API key."""
    return OPENROUTER_API_KEY

def get_wandb_token():
    """Returns the Weights & Biases API key."""
    return WANDB_API_KEY
