# pip install langchain-openai
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

application_prompt = """Given the following short description
    of a particular topic, write 3 attention-grabbing headlines 
    for a blog post. Reply with only the titles, one on each line,
    with no additional text.
    DESCRIPTION:
    {user_input}
"""
user_input = """AI Orchestration with LangChain and LlamaIndex
    keywords: Generative AI, applications, LLM, chatbot"""

llm = ChatOpenAI(
    #base_url="http://localhost:1234/v1",
    temperature=0.7,
    max_tokens=500,
    model='gpt-4-1106-preview'
)
prompt = PromptTemplate(  
    input_variables=["user_input"],
    template=application_prompt
)
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"user_input": user_input})

print(result)

# for streaing use
#results = chain.stream({"user_input": user_input})
#for chunk in results:
#    print(chunk, end='')
