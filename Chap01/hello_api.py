import os
from openai import OpenAI # version 1.0+
# if you get openai errors, run pip install --upgrade openai

llm = OpenAI(
    # place your OpenAI key in an environment variable
    #api_key=os.environ['OPENAI_API_KEY'], # this is the default
    #base_url="http://localhost:1234/v1"  # see chapter 1 video 3
)

system_prompt = """Given the following short description
    of a particular topic, write 3 attention-grabbing headlines 
    for a blog post. Reply with only the titles, one on each line,
    with no additional text.
    DESCRIPTION:
"""
user_input = """AI Orchestration with LangChain and LlamaIndex
    keywords: Generative AI, applications, LLM, chatbot"""

response = llm.chat.completions.create(
    model="gpt-4-1106-preview",
    max_tokens=500,
    temperature=0.7,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
)

print(response.choices[0].message.content)
