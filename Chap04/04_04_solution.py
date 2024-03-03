from langchain.agents import AgentType, Tool, initialize_agent
# pip install langchain-openai
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

#  In order to set this Bing search, go to
# https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
# and click "Try now" to get a Bing web search API key
# put your API key in the env variable BING_SUBSCRIPTION_KEY
from langchain.utilities.bing_search import BingSearchAPIWrapper
# pip install numexpr
from langchain.chains import LLMMathChain

import os
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

llm = ChatOpenAI(
    #base_url="http://localhost:1234/v1/",
    temperature=0
)

tools = [   
    Tool(
        name="Web Search",
        func=BingSearchAPIWrapper().run,
        description="useful for when you need to answer specific questions from information on the web",
    ),
    Tool(
        name="Math Calculator",
        func=LLMMathChain.from_llm(llm),
        description="useful for when you need to perform a mathematical calculation",
    ),
]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

pydict = agent.invoke({"input": "Hi, I am Bob"})
print(pydict["output"])
pydict = agent.invoke({"input": "What's my name?"})
print(pydict["output"])
pydict = agent.invoke({"input": "Who is the CEO of LinkedIn in 2023?"})
print(pydict["output"])

pydict = agent.invoke({"input": "What is 2*pi*10^2"})
print(pydict["output"])
