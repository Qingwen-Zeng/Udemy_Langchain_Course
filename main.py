from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# INPUT:  .env file containing OPENAI_API_KEY=sk-abc123...
# OUTPUT: os.environ["OPENAI_API_KEY"] = "sk-abc123..."
load_dotenv()

#tavily = TavilyClient()  # Initialize Tavily client
"""
# INPUT:  a Python function with docstring + type hints
# OUTPUT: a LangChain Tool object with name="search", description from docstring,
#         and args_schema parsed from the Args section
@tool  # Registers this function as a tool the LLM can call
def search(query: str) -> str:
    # the LLM reads it to decide when to use this tool
    Tools that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search results
    
    
    # INPUT:  query = "Tokyo weather" (passed by the LLM's tool_call)
    # OUTPUT: prints "Searching for: Tokyo weather" to terminal
    print(f"Searching for: {query}")
    return str(tavily.search(query=query))  # Call Tavily API to perform the search and return results
"""


# List of tools the agent can access
tools = [TavilySearch()]  # You can add more tools to this list as needed

# Initialize the LLM 
llm = ChatOpenAI(model="gpt-5")

# INPUT:  LLM object + tools list + system prompt string
# OUTPUT: a compiled LangGraph state machine with two nodes:
#         "agent" (calls LLM) ↔ "tools" (executes functions) in a loop
agent = create_agent(
    model=llm,                                # The LLM to use
    tools=tools,                              # Tools available to the agent
    system_prompt="You are a helpful assistant",  # Prepended to every LLM call
)

def main():
    # INPUT:  {"messages": [HumanMessage("What is the weather in Tokyo?")]}
    #
    # INTERNALLY:
    #   Loop 1 → LLM receives user message → returns AIMessage with tool_calls=[{name:"search", args:{query:"Tokyo weather"}}]
    #   Tool   → executes search("Tokyo weather") → returns ToolMessage("Tokyo weather is sunny")
    #   Loop 2 → LLM receives all messages including tool result → returns AIMessage("The weather in Tokyo is sunny!") → no tool_calls → loop ends
    #
    # OUTPUT: {
    #   "messages": [
    #     HumanMessage("What is the weather in Tokyo?"),
    #     AIMessage(tool_calls=[{name:"search", args:{query:"Tokyo weather"}}]),
    #     ToolMessage("Tokyo weather is sunny"),
    #     AIMessage("The weather in Tokyo is sunny!")
    #   ]
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for 3 job postings for AI engineer using langchain in Copenhagen on linkedin and list their details")]}
    )

    # result = {
    #   "messages": [
    #     HumanMessage(...),    # your input
    #     AIMessage(...),       # LLM decides to call search tool
    #     ToolMessage(...),     # search() result: "Tokyo weather is sunny"
    #     AIMessage(...),       # LLM final answer using the tool result
    #   ]
    # }
    print(result)

    # To get just the final answer:
    # print(result["messages"][-1].content)

if __name__ == "__main__":
    main()