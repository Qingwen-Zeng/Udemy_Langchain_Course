"""
LangChain Agent Schema Examples
================================
Covers:
  Section 2 — Two Strategies (ProviderStrategy vs ToolStrategy)
  Section 3 — Five Schema Types (Pydantic, TypedDict, Dataclass, JSON Schema, Union)

Requirements:
  pip install langchain langchain-openai langchain-tavily pydantic python-dotenv

.env file:
  OPENAI_API_KEY=sk-...
  TAVILY_API_KEY=tvly-...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Literal, Union, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


llm = ChatOpenAI(model="gpt-4o")
tools = [TavilySearch()]


class MovieReview_Provider(BaseModel):
    """A structured movie review."""
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating out of 10", ge=0.0, le=10.0)
    summary: str = Field(description="A 1-2 sentence summary of the review")
    recommend: bool = Field(description="Would you recommend this movie?")


def example_2a_provider_strategy():
    """
    ProviderStrategy — explicitly tells LangChain to use the model
    provider's native structured output API.

    Equivalent shorthand: response_format=MovieReview_Provider
    (LangChain auto-selects ProviderStrategy when the model supports it)
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2a: ProviderStrategy")
    print("=" * 60)

    agent = create_agent(
        model=llm,
        tools=[],  # no tools needed for this example
        response_format=ProviderStrategy(MovieReview_Provider, strict=True),
        #                                                      ^^^^^^^^^^^
        #              strict=True → provider enforces exact schema match
        system_prompt="You are a movie critic. Analyze reviews precisely.",
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            content="Review the movie Inception (2010) by Christopher Nolan"
        )]
    })

    # Access the validated Pydantic instance
    response = result["structured_response"]

    print(f"Title:     {response.title}")
    print(f"Rating:    {response.rating}/10")
    print(f"Summary:   {response.summary}")
    print(f"Recommend: {response.recommend}")
    print(f"Type:      {type(response).__name__}")  # MovieReview_Provider



class MovieReview_Tool(BaseModel):
    """A structured movie review."""
    title: str = Field(description="The movie title")
    rating: int | None = Field(description="Rating from 1-10", ge=1, le=10)
    pros: List[str] = Field(description="List of positive points, 2-5 items")
    cons: List[str] = Field(description="List of negative points, 1-3 items")
    genre: Literal["action", "comedy", "drama", "sci-fi", "horror", "other"] = Field(
        description="Primary genre of the movie"
    )


def example_2b_tool_strategy():
    """
    ToolStrategy — uses tool calling to enforce the schema.
    Supports error handling with automatic retries.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2b: ToolStrategy (with error handling)")
    print("=" * 60)

    agent = create_agent(
        model=llm,
        tools=[],
        response_format=ToolStrategy(
            schema=MovieReview_Tool,
            handle_errors=True,  # auto-retry if validation fails (default)
            tool_message_content="Review captured successfully!",
        ),
        system_prompt="You are a movie critic. Be specific in pros and cons.",
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            content="Review the movie The Matrix (1999)"
        )]
    })

    response = result["structured_response"]

    print(f"Title: {response.title}")
    print(f"Rating: {response.rating}/10")
    print(f"Genre: {response.genre}")
    print(f"Pros: {response.pros}")
    print(f"Cons: {response.cons}")
    print(f"Type: {type(response).__name__}")  # MovieReview_Tool


class Source(BaseModel):
    """A reference source."""
    url: str = Field(description="The URL of the source")
    title: str = Field(description="Title or description of the source")


class ResearchAnswer(BaseModel):
    """A research answer with citations."""
    answer: str = Field(description="Comprehensive answer to the question")
    key_facts: List[str] = Field(description="3-5 key facts extracted from research")
    sources: List[Source] = Field(
        default_factory=list,
        description="Sources used to generate the answer"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="How confident the agent is in the answer"
    )


def example_3a_pydantic():
    """
    Pydantic model — the most powerful schema type.
    Supports nested models, validation constraints, defaults, and Literal types.
    Returns a validated Pydantic instance.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3a: Pydantic Model (with tools + nested models)")
    print("=" * 60)

    agent = create_agent(
        model=llm,
        tools=tools,  # TavilySearch for web research
        response_format=ResearchAnswer,
        system_prompt="You are a research assistant. Always cite sources with URLs.",
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            content="What is the ReAct prompting framework for LLMs?"
        )]
    })

    response = result["structured_response"]

    print(f"Answer:     {response.answer[:100]}...")
    print(f"Confidence: {response.confidence}")
    print(f"Key facts:")
    for fact in response.key_facts:
        print(f"  - {fact}")
    print(f"Sources:")
    for src in response.sources:
        print(f"  - {src.title}: {src.url}")

    # Pydantic-specific features:
    print(f"\nAs dict: {json.dumps(response.model_dump(), indent=2)[:200]}...")
    print(f"Type:    {type(response).__name__}")  # ResearchAnswer


class WeatherInfo(TypedDict):
    city: str
    temperature: str
    condition: str
    recommendation: str


def example_3b_typeddict():
    """
    TypedDict — lightweight alternative to Pydantic.
    No validation constraints (no ge=, le=, Literal).
    Returns a plain dict.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3b: TypedDict")
    print("=" * 60)

    agent = create_agent(
        model=llm,
        tools=tools,
        response_format=WeatherInfo,
        system_prompt="You are a weather assistant. Search for current weather data.",
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            content="What is the weather like in Copenhagen right now?"
        )]
    })

    response = result["structured_response"]

    # response is a plain dict, not a class instance
    print(f"City:           {response['city']}")
    print(f"Temperature:    {response['temperature']}")
    print(f"Condition:      {response['condition']}")
    print(f"Recommendation: {response['recommendation']}")
    print(f"Type:           {type(response).__name__}")  # dict


@dataclass
class BookSummary:
    title: str
    author: str
    genre: str
    summary: str
    page_count_estimate: int


def example_3c_dataclass():
    """
    Dataclass — another lightweight option.
    Returns a dict (NOT a dataclass instance).
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3c: Dataclass")
    print("=" * 60)

    agent = create_agent(
        model=llm,
        tools=[],
        response_format=BookSummary,
        system_prompt="You are a book expert. Provide accurate book information.",
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            content="Tell me about the book '1984' by George Orwell"
        )]
    })

    response = result["structured_response"]

    # response is a dict
    print(f"Title:  {response['title']}")
    print(f"Author: {response['author']}")
    print(f"Genre:  {response['genre']}")
    print(f"Summary: {response['summary'][:100]}...")
    print(f"Pages:  ~{response['page_count_estimate']}")
    print(f"Type:   {type(response).__name__}")  # dict


RECIPE_SCHEMA = {
    "type": "object",
    "properties": {
        "dish_name": {
            "type": "string",
            "description": "Name of the dish",
        },
        "cuisine": {
            "type": "string",
            "description": "Type of cuisine (e.g., Italian, Japanese)",
        },
        "ingredients": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of ingredients needed",
        },
        "prep_time_minutes": {
            "type": "integer",
            "description": "Preparation time in minutes",
        },
    },
    "required": ["dish_name", "cuisine", "ingredients", "prep_time_minutes"],
}


def example_3d_json_schema():
    """
    JSON Schema — use when you don't want to define a Python class.
    Useful for dynamic schemas or schemas loaded from config files.
    Returns a dict.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3d: JSON Schema (raw dict)")
    print("=" * 60)

    agent = create_agent(
        model=llm,
        tools=[],
        response_format=RECIPE_SCHEMA,
        system_prompt="You are a chef. Suggest recipes based on user requests.",
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            content="Suggest a quick Italian pasta dish"
        )]
    })

    response = result["structured_response"]

    print(f"Dish:        {response['dish_name']}")
    print(f"Cuisine:     {response['cuisine']}")
    print(f"Prep time:   {response['prep_time_minutes']} minutes")
    print(f"Ingredients: {response['ingredients']}")
    print(f"Type:        {type(response).__name__}")  # dict


# The model chooses which schema fits the input best.
# MUST use ToolStrategy — ProviderStrategy does not support Union.

class PersonInfo(BaseModel):
    """Information about a person."""
    name: str = Field(description="Full name")
    role: str = Field(description="Job title or role")
    email: str | None = Field(default=None, description="Email if available")


class CompanyInfo(BaseModel):
    """Information about a company."""
    name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    headquarters: str = Field(description="HQ location")
    founded_year: int | None = Field(default=None, description="Year founded")


class ProductInfo(BaseModel):
    """Information about a product."""
    name: str = Field(description="Product name")
    category: str = Field(description="Product category")
    price: str | None = Field(default=None, description="Price if available")
    description: str = Field(description="Brief product description")


def example_3e_union_types():
    """
    Union types — the model picks whichever schema best fits the input.
    Only works with ToolStrategy.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3e: Union Types (model picks the schema)")
    print("=" * 60)

    agent = create_agent(
        model=llm,
        tools=tools,
        response_format=ToolStrategy(
            schema=Union[PersonInfo, CompanyInfo, ProductInfo],
            handle_errors=True,
        ),
        system_prompt=(
            "You are an information extraction assistant. "
            "Extract the most relevant entity from the user's query."
        ),
    )

    # --- Test 1: Should pick PersonInfo ---
    print("\n--- Query about a person ---")
    result1 = agent.invoke({
        "messages": [HumanMessage(
            content="Tell me about Jensen Huang, CEO of NVIDIA"
        )]
    })
    r1 = result1["structured_response"]
    print(f"Schema chosen: {type(r1).__name__}")
    print(f"Data: {r1}")

    # --- Test 2: Should pick CompanyInfo ---
    print("\n--- Query about a company ---")
    result2 = agent.invoke({
        "messages": [HumanMessage(
            content="Tell me about Anthropic, the AI safety company"
        )]
    })
    r2 = result2["structured_response"]
    print(f"Schema chosen: {type(r2).__name__}")
    print(f"Data: {r2}")

    # --- Test 3: Should pick ProductInfo ---
    print("\n--- Query about a product ---")
    result3 = agent.invoke({
        "messages": [HumanMessage(
            content="Tell me about the iPhone 16 Pro"
        )]
    })
    r3 = result3["structured_response"]
    print(f"Schema chosen: {type(r3).__name__}")
    print(f"Data: {r3}")


def print_comparison():
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("""
┌──────────────────┬───────────────┬──────────────┬────────────────┐
│ Schema Type      │ Returns       │ Dot Access?  │ Validation?    │
├──────────────────┼───────────────┼──────────────┼────────────────┤
│ Pydantic Model   │ Pydantic obj  │ Yes (.title) │ Full (ge, le..)│
│ TypedDict        │ dict          │ No  ["title"]│ Type hints only│
│ Dataclass        │ dict          │ No  ["title"]│ Type hints only│
│ JSON Schema      │ dict          │ No  ["title"]│ JSON Schema    │
│ Union[A, B]      │ A or B inst.  │ Yes (.title) │ Per-model      │
└──────────────────┴───────────────┴──────────────┴────────────────┘

┌──────────────────┬───────────────┬──────────────┬────────────────┐
│ Strategy         │ Reliability   │ Union Types? │ Error Retry?   │
├──────────────────┼───────────────┼──────────────┼────────────────┤
│ ProviderStrategy │ Highest       │ No           │ No             │
│ ToolStrategy     │ High          │ Yes          │ Yes            │
│ Auto (default)   │ Best available│ Depends      │ Depends        │
└──────────────────┴───────────────┴──────────────┴────────────────┘
""")

if __name__ == "__main__":
    print("LangChain Agent Schema Examples")
    print("Choose which example to run:\n")
    print("  2a  ProviderStrategy")
    print("  2b  ToolStrategy (with error handling)")
    print("  3a  Pydantic Model (with tools + nested models)")
    print("  3b  TypedDict")
    print("  3c  Dataclass")
    print("  3d  JSON Schema")
    print("  3e  Union Types")
    print("  all Run all examples")
    print("  cmp Print comparison table")

    choice = input("\nEnter choice: ").strip().lower()

    examples = {
        "2a": example_2a_provider_strategy,
        "2b": example_2b_tool_strategy,
        "3a": example_3a_pydantic,
        "3b": example_3b_typeddict,
        "3c": example_3c_dataclass,
        "3d": example_3d_json_schema,
        "3e": example_3e_union_types,
        "cmp": print_comparison,
    }

    if choice == "all":
        for fn in examples.values():
            fn()
    elif choice in examples:
        examples[choice]()
    else:
        print(f"Unknown choice: {choice}")