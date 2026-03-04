from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langsmith import traceable
load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# ---- Define the tool ----
@tool
def get_product_price(product: str) -> float:
    """Get the price of a product."""
    print(f"Executing get_product_price for {product}...")
    prices = {
        "laptop": 1299.99,
        "headphones": 199.99,
        "keyboard": 89.99,
    }
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price.
    Available discount tiers: bronze, silver, gold
    """
    print(f"Executing apply_discount for price {price} with discount tier {discount_tier}...")
    discount_precentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_precentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# ---- agent loop  ----
@traceable(name="langchain_agent_loop")
def run_agent(question:str):
    pass


if __name__ == "__main__":
    print("Hello LangChain Agent")
    print("Ask a question about product prices and discounts.")
    result = run_agent("What is the price of a laptop with a gold discount?")