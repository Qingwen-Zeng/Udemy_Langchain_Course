# ============================================================
# ReAct Agent using Ollama + Raw Text Parsing (No LangChain)
# ============================================================
# This file implements a ReAct (Reason + Act) agent loop from scratch.
# Instead of using LangChain's agent framework, it uses:
#   - A single prompt string to guide the LLM
#   - Regex to parse the LLM's text output
#   - Plain Python functions as tools
# ============================================================

# STEP 1: Import required libraries
# ----------------------------------
# re      → for regex parsing of LLM output (Action, Action Input, Final Answer)
# inspect → for auto-extracting function signatures and docstrings
import re
import inspect

# Load environment variables from .env file
# (e.g., LANGSMITH_API_KEY, OPENAI_API_KEY)
from dotenv import load_dotenv
load_dotenv()

# ollama  → runs the LLM locally on your machine
# traceable → LangSmith decorator to log/trace function calls for observability
import ollama
from langsmith import traceable

# STEP 2: Define agent-level constants
# --------------------------------------
MAX_ITERATIONS = 10          # Safety cap — agent stops after 10 loops even without a final answer
MODEL = "qwen3:1.7b"         # The local Ollama model to use for reasoning


# ============================================================
# STEP 3: Define the Tools (plain Python functions)
# ============================================================
# These are the actual tools the agent can call.
# @traceable logs every call to LangSmith for debugging/monitoring.
# The docstring is important — it will be auto-extracted into the prompt
# so the LLM knows what each tool does.

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    # Debug print so you can see in the terminal when a tool is executing
    print(f"    >> Executing get_product_price(product='{product}')")

    # Simulated product catalog (in a real system, this would query a database)
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}

    # Return price if found, or 0 if product doesn't exist
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")

    # Ensure price is a float (LLM might pass it as a string)
    price = float(price)

    # Discount lookup table — maps tier name → percentage off
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}

    # Get discount %, default to 0 if tier is unknown
    discount = discount_percentages.get(discount_tier, 0)

    # Calculate and return the discounted price, rounded to 2 decimal places
    return round(price * (1 - discount / 100), 2)


# STEP 4: Register all tools in a dict
# --------------------------------------
# Key   = tool name string (must match what the LLM will write in "Action:")
# Value = the actual Python function to call
tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# ============================================================
# STEP 5: Auto-generate tool descriptions for the prompt
# ============================================================
# Instead of manually writing JSON schemas, we use Python's inspect
# module to pull the function signature + docstring automatically.
# This keeps tool definitions as the single source of truth.

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():

        # @traceable wraps the function and adds extra params like config=None
        # __wrapped__ lets us access the original, unwrapped function
        # so we get the correct signature the LLM should see
        original_function = getattr(tool_function, "__wrapped__", tool_function)

        # Extract the function signature e.g. (product: str) -> float
        signature = inspect.signature(original_function)

        # Extract the docstring (the description between triple quotes)
        docstring = inspect.getdoc(tool_function) or ""

        # Format as: tool_name(args) - description
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    # Join all tool descriptions with newlines for the prompt
    return "\n".join(descriptions)


# Generate the formatted tool descriptions string
tool_descriptions = get_tool_descriptions(tools)

# Generate a comma-separated list of tool names (for the prompt's [tool_names] slot)
tool_names = ", ".join(tools.keys())


# ============================================================
# STEP 6: Build the ReAct Prompt Template
# ============================================================
# This prompt IS the agent's "brain". It tells the LLM:
#   1. What tools exist and their signatures
#   2. The exact text format to follow (Thought/Action/Observation)
#   3. Hard rules to prevent hallucination
# {question} is the only runtime variable — filled in run_agent().

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""
# Note: {{question}} uses double braces to escape the f-string,
# so it stays as a literal {question} placeholder for .format() later


# ============================================================
# STEP 7: Traceable wrapper for the Ollama LLM call
# ============================================================
# We wrap ollama.chat() with @traceable so LangSmith logs
# every LLM call (model, messages, options, response).

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


# ============================================================
# STEP 8: The Main Agent Loop
# ============================================================
# This function runs the full ReAct loop:
#   - Sends prompt to LLM
#   - Parses output for Final Answer or tool call
#   - Executes tool if needed
#   - Appends result to scratchpad and repeats

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # STEP 8a: Format the base prompt with the user's question
    # The scratchpad starts empty and grows with each iteration
    prompt = react_prompt.format(question=question)
    scratchpad = ""   # Will accumulate Thought/Action/Observation history as a string

    # STEP 8b: Start the ReAct loop (max MAX_ITERATIONS times)
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Combine the base prompt with the accumulated history
        # This gives the LLM full context of what has happened so far
        full_prompt = prompt + scratchpad

        # STEP 8c: Call the LLM
        # Key option: stop=["\nObservation"] tells the LLM to STOP generating
        # as soon as it writes a newline before "Observation".
        # This prevents it from hallucinating fake tool results.
        # temperature=0 makes output deterministic (no randomness)
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )

        # Extract the text content from the LLM response
        output = response.message.content
        print(f"LLM Output:\n{output}")

        # --------------------------------------------------------
        # STEP 8d: Check if the LLM has produced a Final Answer
        # --------------------------------------------------------
        # Regex looks for "Final Answer: <anything>" in the output
        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            # Extract just the answer text (group 1 = everything after "Final Answer: ")
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer   # ← Exit the loop and return the result

        # --------------------------------------------------------
        # STEP 8e: No Final Answer — check for a tool call instead
        # --------------------------------------------------------
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        # Regex looks for "Action: <tool_name>"
        action_match = re.search(r"Action:\s*(.+)", output)

        # Regex looks for "Action Input: <args>"
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        # If either is missing, the LLM didn't follow the format — abort
        if not action_match or not action_input_match:
            print("  [Parsing] ERROR: Could not parse Action/Action Input from LLM output")
            break

        # Extract tool name and raw argument string from regex matches
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # --------------------------------------------------------
        # STEP 8f: Parse the tool arguments
        # --------------------------------------------------------
        # Split on commas: "1299.99, gold" → ["1299.99", "gold"]
        raw_args = [x.strip() for x in tool_input_raw.split(",")]

        # Strip key=value prefix if present: "price=1299.99" → "1299.99"
        # Also strip surrounding quotes: "'laptop'" → "laptop"
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        # --------------------------------------------------------
        # STEP 8g: Execute the tool
        # --------------------------------------------------------
        print(f"  [Tool Executing] {tool_name}({args})...")

        if tool_name not in tools:
            # Tool name didn't match — give the LLM a helpful error message
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            # Call the actual Python function with the parsed arguments
            # *args unpacks the list as positional arguments
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        # --------------------------------------------------------
        # STEP 8h: Append this iteration's result to the scratchpad
        # --------------------------------------------------------
        # The scratchpad grows like this after each iteration:
        #
        #   Thought: I need to get the price first
        #   Action: get_product_price
        #   Action Input: laptop
        #   Observation: 1299.99
        #   Thought: <next iteration starts here>
        #
        # This entire history is re-sent to the LLM on the next iteration
        # so it remembers what it already did.
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    # If we exit the loop without a Final Answer, something went wrong
    print("ERROR: Max iterations reached without a final answer")
    return None


# ============================================================
# STEP 9: Entry point — run the agent with a sample question
# ============================================================
if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()

    # This will trigger two tool calls:
    #   1. get_product_price("laptop")    → 1299.99
    #   2. apply_discount(1299.99, "gold") → 999.99
    # Then the LLM will produce: Final Answer: The laptop costs $999.99
    result = run_agent("What is the price of a laptop after applying a gold discount?")