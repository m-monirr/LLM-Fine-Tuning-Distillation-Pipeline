"""Pydantic models for data validation and schema definition."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


StoryCategory = Literal[
    "politics", "sports", "art", "technology", "economy",
    "health", "entertainment", "science", "not_specified"
]

EntityType = Literal[
    "person-male", "person-female", "location", "organization", 
    "event", "time", "quantity", "money", "product", "law", 
    "disease", "artifact", "not_specified"
]


class Entity(BaseModel):
    """Entity extracted from news story."""
    entity_value: str = Field(..., description="The actual name or value of the entity.")
    entity_type: EntityType = Field(..., description="The type of recognized entity.")


class NewsDetails(BaseModel):
    """Structured details extracted from Arabic news story."""
    story_title: str = Field(
        ..., 
        min_length=5, 
        max_length=300,
        description="A fully informative and SEO optimized title of the story."
    )
    story_keywords: List[str] = Field(
        ..., 
        min_items=1,
        description="Relevant keywords associated with the story."
    )
    story_summary: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=5,
        description="Summarized key points about the story (1-5 points)."
    )
    story_category: StoryCategory = Field(
        ..., 
        description="Category of the news story."
    )
    story_entities: List[Entity] = Field(
        ..., 
        min_items=1, 
        max_items=10,
        description="List of identified entities in the story."
    )


class TranslatedStory(BaseModel):
    """Translation of Arabic news story."""
    translated_title: str = Field(
        ..., 
        min_length=5, 
        max_length=300,
        description="Suggested translated title of the news story."
    )
    translated_content: str = Field(
        ..., 
        min_length=5,
        description="Translated content of the news story."
    )


class SFTRecord(BaseModel):
    """Record in supervised finetuning dataset."""
    id: int
    story: str
    task: str
    output_scheme: str
    status: Literal["success", "failed"]
    response: Union[Dict[str, Any], str]
