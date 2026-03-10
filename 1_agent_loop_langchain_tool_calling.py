from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langsmith import traceable
load_dotenv()
"""
@tool auto-generates a tool wrapper for a function, allowing the LLM to call it as a tool.
init_chat_model(...).bind_tools(...) binds the tools to the LLM, allowing it to call them during generation.
tool_calls is a property of AIMessage that contains any tool calls the LLM made in that message.
"""

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
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"User question: {question}")
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]
    
    for iterration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iterration} ---")
        # send the conversation history to the LLM, which will return an AIMessage that may contain tool calls
        ai_message = llm_with_tools.invoke(messages)
        # extract the tool calls from the AIMessage
        tool_calls = ai_message.tool_calls
        
        #if no tool calls, we assume the agent is done and has returned a final answer
        if not tool_calls:
            print(f"\nFinal answer: {ai_message.content}")
            return ai_message.content  # Final answer, no more tool calls
        
        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        # tool_call_id is a unique ID the LLM generates for each tool call request,
        #ToolMessage needs it so the LLM can match the result back to the request it made. 
        """
        The conversation history looks like this:
            HumanMessage:   "What is the price of a laptop after gold discount?"
            AIMessage:      content=""
                            tool_calls=[{
                                "id": "call_abc123",       ← LLM generates this
                                "name": "get_product_price",
                                "args": {"product": "laptop"}
                            }]
            ToolMessage:    content="1299.99"
                            tool_call_id="call_abc123"     ← must match 
        """
        tool_call_id = tool_call.get("id")
        
        print(f"LLM called tool: {tool_name} with args: {tool_args}")
        # Find the tool function to call based on the tool name
        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"LLM called unknown tool: {tool_name}")
        # Call the tool function with the provided arguments and get the observation
        observation = tool_to_use.invoke(tool_args)
        print(f"Tool observation: {observation}")
        
        messages.append(ai_message)  # Add the AI message with the tool call
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )  # Add the tool observation as a ToolMessage
    print("Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")