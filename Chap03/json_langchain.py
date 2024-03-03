# pip install langchain-openai
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable.config import RunnableConfig

prompt = """Write a weather report for a major city
    in ten words or less.
    Do not include any additional explanation.
"""

guided_prompt = prompt + """
Return the result as JSON as follows:
{ "city": "<CITY_NAME>",
"report": "<BRIEF_REPORT>" }
"""

chat = ChatOpenAI(
    #base_url="http://localhost:1234/v1",
    temperature=0.7,
    max_tokens=500,
    model='gpt-4-1106-preview'
)

def baseline():
    print("baseline:")
    print(chat.invoke(prompt).content)

def with_guided_prompt():
    print("1. Ask nicely")
    print(chat.invoke(guided_prompt).content)

def with_pydantic_output_formatter():
    print("2. Pydantic OutputParser")
    from langchain.output_parsers import PydanticOutputParser
    from langchain.pydantic_v1 import BaseModel, Field

    class WeatherReport(BaseModel):
        city: str = Field(description="City name")
        report: str = Field(description="Brief weather report")

    parser = PydanticOutputParser(pydantic_object=WeatherReport)
    #print(f"Parser instructions: {parser.get_format_instructions()}")

    runnable_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=parser.get_format_instructions()),
            HumanMessage(content=prompt)
        ]
    )
    chain = runnable_prompt | chat | parser
    py_obj = chain.invoke({})
    print(py_obj.city, py_obj.report)

    #To be extra robust about fixing JSON errors you can add
    #from langchain.output_parsers import OutputFixingParser
    #robust_parser = OutputFixingParser.from_llm(
    #    parser=parser,
    #    llm=chat
    #)
    # this will re-prompt to get a conforming output format


if __name__ == "__main__":
    baseline()
    with_guided_prompt()
    with_pydantic_output_formatter()
