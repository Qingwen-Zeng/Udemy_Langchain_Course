from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
load_dotenv()  # Load environment variables from .env file


def main():
    print("Hello from udemy-langchain-course!")
    # print(os.environ.get('OPENAI_API_KEY'))
    
    # Input: str (raw text)
    # Output: str (stored in variable)
    information = """  
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of February 2026, Forbes estimates his net worth to be around US$852 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals.

Musk was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated as president in early 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). After a public feud with Trump, Musk left the Trump administration and returned to managing his companies. Musk is a supporter of global far-right figures, causes, and political parties. His political activities, views, and statements have made him a polarizing figure. Musk has been criticized for COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service, following his pledge to decrease censorship. His role in the second Trump administration attracted public backlash, particularly in response to DOGE. The emails he sent to Jeffrey Epstein are included in the Epstein files, which were published between 2025–26 and became a topic of worldwide debate.
    """
    # Input: str (template string with {information} placeholder)
    # Output: str (stored in variable)
    summary_template ="""
    given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about the person
    """
    
    # Input: List[str] (input_variables), str (template)
    # Output: PromptTemplate (a Runnable that can format prompts)
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    
    # Input: str (model name), float (temperature)
    # Output: ChatOpenAI (a Runnable that wraps the OpenAI chat API)
    llm = ChatOpenAI(temperature=0, model="gpt-5") # samll temperature means the model will be more deterministic in its responses
    # llm = ChatOllama(temperature=0.7, model="gemma3:270m") # using ollama to run a local model, we can specify the model name and temperature, and it will return a ChatOllama instance that we can use to invoke the model
    
    # Input: PromptTemplate, ChatOpenAI
    # Output: RunnableSequence (via __or__ / pipe operator)
    chain = summary_prompt_template | llm # this is called chaining, we are chaining the prompt template with the language model, so that the output of the prompt template will be the input of the language model
    """
    # Which is equivalent to
    chain = RunnableSequence(summary_prompt_template, llm)
    """
    
    # Input: Dict[str, str] → {"information": "Elon Reeve Musk..."}
    #   Step 1: PromptTemplate.invoke(Dict)    → PromptValue
    #   Step 2: PromptValue.to_messages()      → List[HumanMessage]
    #   Step 3: ChatOpenAI.invoke(List[HumanMessage]) → AIMessage
    # Output: AIMessage (contains .content with the LLM's response text)
    response = chain.invoke(input={"information": information}) # chain is a runnable interface, we can invoke it with the input, and it will return the output of the language model
    print(response.content) # response is an AIMessage, we can access the content of the message to get the text response from the language model
    
    
if __name__ == "__main__":
    main()
