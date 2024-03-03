from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
# pip install llama-index-llms-openai-like
from llama_index.llms.openai_like import OpenAILike
# pip install llama-index-tools-bing-search
# In order to set up Bing search, go to
# https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
# and click "Try now" to get a Bing web search API key
# put your API key in the env variable BING_SUBSCRIPTION_KEY
from llama_index.tools.bing_search import BingSearchToolSpec

import os
api_key = os.environ["BING_SUBSCRIPTION_KEY"]
tool_spec = BingSearchToolSpec(api_key=api_key)
tool_list = tool_spec.to_tool_list()

llm = OpenAILike(
    is_chat_model=True,
    model="gpt-4-1106-preview",
    #api_base="http://localhost:1234/v1/"
)
agent = ReActAgent.from_tools(tool_list, llm=llm, verbose=True)

response = agent.chat("Hi, I am Bob.")
print(str(response))

response = agent.chat("What's my name?")
print(str(response))

response = agent.chat("Who is the CEO of LinkedIn in 2023?")
print(str(response))
