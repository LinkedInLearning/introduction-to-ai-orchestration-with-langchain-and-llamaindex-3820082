# pip install llama-index-agent-openai
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
import json
import random

def get_weather_for_city(city):
    """Get the current weather in a given city"""
    print(f"Calling local get_weather_for_city for {city}")
    return json.dumps({"city": city, "temperature": random.randint(1,50)})

llm = OpenAI(model="gpt-3.5-turbo-1106")
tool = FunctionTool.from_defaults(fn=get_weather_for_city)
agent = OpenAIAgent.from_tools([tool], llm=llm, verbose=True)
response = agent.chat(
    "What's the weather like in Miami?"
)

print(response)
