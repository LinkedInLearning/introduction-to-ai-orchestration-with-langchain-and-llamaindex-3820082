# pip install llama-index-llms-openai-like
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage

prompt = ChatMessage(
    role="user",
    content="""Write a weather report for a random city
        in ten words or less.
        Do not include any additional explanation.
""")

guided_prompt = ChatMessage(role="user", content=prompt.content + """
Return the result as JSON as follows:
{ "city": "<CITY_NAME>",
"report": "<SHORT_REPORT>" }
""")

chat = OpenAILike(
    is_chat_model=True,
    is_function_calling_model=True,
    #api_base="http://localhost:1234/v1",
    temperature=0.7,
    max_tokens=500,
    model="gpt-4-1106-preview",
)

def baseline():
    print("baseline:")
    print(chat.chat([prompt]))

def with_guided_prompt():
    print("1. Ask nicely")
    print(chat.chat([guided_prompt]))

def with_openai_pydantic():
    print("2. OpenAIPydandicProgram")
    from pydantic import BaseModel, Field
    from llama_index.program.openai import OpenAIPydanticProgram

    class WeatherReport(BaseModel):
        "A concise weather report for a single city"
        city: str = Field(description="City name")
        report: str = Field(description="Brief weather report")

    program = OpenAIPydanticProgram.from_defaults(
        llm=chat,
        output_cls=WeatherReport,
        prompt_template_str=prompt.content,
        verbose=True,
    )
    print(guided_prompt.content)
    py_obj = program()
    # Now a standard python obj
    print(py_obj.city, py_obj.report)


if __name__ == "__main__":
    baseline()
    with_guided_prompt()
    with_openai_pydantic()

