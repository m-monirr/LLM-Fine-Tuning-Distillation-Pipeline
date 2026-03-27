"""
Generate SFT data using knowledge distillation from larger models via OpenRouter API.
"""

import json
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import requests
from tqdm.auto import tqdm
import json_repair
from dotenv import load_dotenv

from .models import NewsDetails, TranslatedStory, SFTRecord

# Load environment variables
load_dotenv()


class DataGenerator:
    """Generate training data using OpenRouter API."""

    OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
    
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
            parsed = json_repair.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    
    def _fix_json(self, text: str) -> Optional[dict]:
        """Attempt to clean and parse JSON from text response."""
        try:
            text = text.strip()
            if text.startswith("{") and text.endswith("}"):
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else None
            else:
                return self._parse_json(text)
        except Exception:
            return None

    def _call_openrouter_api(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call OpenRouter Chat Completions API and return assistant text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        response = requests.post(
            self.OPENROUTER_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from exc

    def _chat_json(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """Helper to get a JSON object from the model (best-effort)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self._call_openrouter_api(messages)
        return self._fix_json(text)
    
    def generate_data(self, prompts: List[str]) -> List[dict]:
        """Generate JSON objects for the given prompts (best-effort)."""
        results = []
        for prompt in tqdm(prompts, desc="Generating data"):
            try:
                text = self._call_openrouter_api(
                    messages=[
                        {
                            "role": "system",
                            "content": "Return only a single valid JSON object. No markdown, no extra text.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
                json_data = self._fix_json(text)
                if json_data:
                    results.append(json_data)
                else:
                    print(f"Warning: Failed to parse JSON from response: {text}")
            except Exception as e:
                print(f"Error processing prompt '{prompt}': {e}")
        return results

    def generate_news_details(self, headlines: List[str]) -> List[NewsDetails]:
        """Generate structured `NewsDetails` for the given headlines (best-effort)."""
        prompts = [
            "\n".join(
                [
                    "Extract structured news details as JSON.",
                    "Return only JSON with keys: story_title, story_keywords, story_summary, story_category, story_entities.",
                    f"Headline: {headline}",
                ]
            )
            for headline in headlines
        ]
        data = self.generate_data(prompts)
        validated: List[NewsDetails] = []
        for item in data:
            try:
                validated.append(NewsDetails(**item))
            except Exception:
                continue
        return validated

    def generate_translated_stories(self, stories: List[str], target_language: str) -> List[TranslatedStory]:
        """Generate translated stories for the given stories (best-effort)."""
        prompts = [
            "\n".join(
                [
                    "Translate the following Arabic news story.",
                    f"Target language: {target_language}",
                    "Return only JSON with keys: translated_title, translated_content.",
                    "Story:",
                    story,
                ]
            )
            for story in stories
        ]
        data = self.generate_data(prompts)
        validated: List[TranslatedStory] = []
        for item in data:
            try:
                validated.append(TranslatedStory(**item))
            except Exception:
                continue
        return validated

    def _extract_details_json(self, story: str) -> Optional[dict]:
        system_prompt = (
            "You label Arabic news stories. "
            "Return only a single valid JSON object (no markdown, no commentary)."
        )
        user_prompt = "\n".join(
            [
                "Task: Extract structured details from the Arabic story.",
                "Output must be a JSON object with EXACT keys:",
                "- story_title (string)",
                "- story_keywords (array of strings)",
                "- story_summary (array of 1-5 strings)",
                "- story_category (one of: politics, sports, art, technology, economy, health, entertainment, science, not_specified)",
                "- story_entities (array of 1-10 objects with keys: entity_value, entity_type)",
                "Entity entity_type must be one of: person-male, person-female, location, organization, event, time, quantity, money, product, law, disease, artifact, not_specified",
                "If unsure, use not_specified.",
                "Story:",
                story,
            ]
        )

        parsed = self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if not parsed:
            return None

        try:
            validated = NewsDetails(**parsed)
            return validated.model_dump()
        except Exception:
            return None

    def _translate_story_json(self, story: str, target_language: str) -> Optional[dict]:
        system_prompt = (
            "You translate Arabic news. "
            "Return only a single valid JSON object (no markdown, no commentary)."
        )
        user_prompt = "\n".join(
            [
                f"Task: Translate the story to {target_language}.",
                "Output must be a JSON object with EXACT keys:",
                "- translated_title (string)",
                "- translated_content (string)",
                "Story:",
                story,
            ]
        )

        parsed = self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if not parsed:
            return None

        try:
            validated = TranslatedStory(**parsed)
            return validated.model_dump()
        except Exception:
            return None

    @staticmethod
    def _story_from_raw_item(item: Dict[str, Any]) -> Optional[str]:
        """Best-effort extraction of story text from a raw JSONL record."""
        content = item.get("content") or item.get("text") or item.get("body")
        if not content:
            return None

        title = item.get("title")
        description = item.get("description")
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if description:
            parts.append(f"Description: {description}")
        parts.append(str(content))
        return "\n\n".join(parts).strip()

    def generate_sft_dataset(
        self,
        raw_data: list,
        output_path: str,
        include_translations: bool = True,
        target_languages: Optional[list] = None,
        max_samples: int = None,
    ) -> None:
        """Generate an SFT JSONL dataset compatible with `DataLoader.load_sft_data`."""

        if target_languages is None:
            target_languages = ["English"]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        items = raw_data[:max_samples] if max_samples is not None else raw_data

        records: List[SFTRecord] = []
        next_id = 0

        extraction_success = 0
        translation_success = 0

        for item in tqdm(items, desc="Processing articles"):
            story = self._story_from_raw_item(item)
            if not story:
                continue

            details = None
            try:
                details = self._extract_details_json(story)
            except Exception:
                details = None

            if details is not None:
                extraction_success += 1
                records.append(
                    SFTRecord(
                        id=next_id,
                        story=story,
                        task="extract_news_details",
                        output_scheme="NewsDetails",
                        status="success",
                        response=details,
                    )
                )
            else:
                records.append(
                    SFTRecord(
                        id=next_id,
                        story=story,
                        task="extract_news_details",
                        output_scheme="NewsDetails",
                        status="failed",
                        response="",
                    )
                )
            next_id += 1

            if include_translations:
                for language in target_languages:
                    translated = None
                    try:
                        translated = self._translate_story_json(story, str(language))
                    except Exception:
                        translated = None

                    task_name = f"translate_to_{str(language).lower()}"
                    if translated is not None:
                        translation_success += 1
                        records.append(
                            SFTRecord(
                                id=next_id,
                                story=story,
                                task=task_name,
                                output_scheme="TranslatedStory",
                                status="success",
                                response=translated,
                            )
                        )
                    else:
                        records.append(
                            SFTRecord(
                                id=next_id,
                                story=story,
                                task=task_name,
                                output_scheme="TranslatedStory",
                                status="failed",
                                response="",
                            )
                        )
                    next_id += 1

        with output_file.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

        total = len(records)
        successful = sum(1 for r in records if r.status == "success")
        failed = total - successful
        print(f"✅ Generated {extraction_success} extraction examples")
        if include_translations:
            print(f"✅ Generated {translation_success} translation examples")
        print(f"✅ Total: {total} training examples")
        if total > 0:
            print(f"✅ Success rate: {successful / total * 100:.1f}%")
        print(f"📁 Saved to: {output_file}")