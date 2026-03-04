from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# INPUT:  .env file containing OPENAI_API_KEY=sk-abc123...
# OUTPUT: os.environ["OPENAI_API_KEY"] = "sk-abc123..."
load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )
    
# Initialize the LLM 
llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [TavilySearch()]

# INPUT:  LLM object + tools list + system prompt string
# OUTPUT: a compiled LangGraph state machine with two nodes:
#         "agent" (calls LLM) ↔ "tools" (executes functions) in a loop
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


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
        {"messages": [HumanMessage(
            content="Search for 3 job postings for AI engineer using langchain in Copenhagen on linkedin and list their details")]}
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