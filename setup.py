"""
Setup script for Arabic News Assistant package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="arabic-news-assistant",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Finetuning pipeline for Arabic news processing with Qwen models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/arabic-news-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "unsloth",
        "pandas>=2.0.0",
        "numpy<2.0.0",
        "pydantic>=2.0.0",
        "json-repair>=0.7.0",
        "openai>=1.0.0",
        "requests>=2.31.0",
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "optional": [
            "faker>=20.0.0",
            "optimum>=1.14.0",
            "vllm>=0.2.0",
        ],
    },
)
