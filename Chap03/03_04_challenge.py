import datetime
import random
from langchain.chains import ConversationChain
# pip install langchain-openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# task functions
def get_current_time():
    """Get the current time

    Returns:
        str: A date and time formatted as YYYY-mm-DDTHH:MM
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%dT%H:%M")

def get_weather(city: str):
    """Get the current weather in a given city

    Args:
        city (str): The city name, e.g. San Francisco

    Returns:
        str: A JSON formatted report with fields 'city' and 'temperature'
    """
    print(f"Calling local get_weather for {city}")
    return f"The weather in {city} is {random.randint(1,50)}C."

# task offloading prompt
system_prompt = """You are a helpful assistant. If asked something you don't know for sure,
of if you are requested to perform an action you are not otherwise capable of,
YOU may choose to respond with one of the following commands, which will get fulfilled
in a subsequent conversational turn:

"#TASK:TIME" to request the current time and date in a format like 2024-01-01T12:34

Do not include any additional explanation when replying with one of these commands.
Upon recieving a response from a command, rephrase it in friendly language,
without adding any additional explanation.

Current conversation:
{history}

Human: {input}
AI:"""


def main():
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        verbose=True,
        temperature=0.6
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
        ]
    )

    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )

    #Start REPL loop
    while True:
        user_input = input("Ask a question. Type 'exit' to quit.\n>")
        if user_input=="exit":
            break
        result = conversation.invoke({"input": user_input})
        print(result)
        response = result["response"].strip()
        print("AI:", response)
        if response.startswith("#TASK:"):
            print("got a task offloading request...")
            # task offloading


if __name__ == "__main__":
    main()