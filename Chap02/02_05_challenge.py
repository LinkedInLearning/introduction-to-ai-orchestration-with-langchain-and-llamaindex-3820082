# pip install llama-index-llms-openai-like
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage
from llama_index.core.readers import SimpleDirectoryReader

application_prompt = """<insert summarization prompt here>

    DOCUMENT:
"""

llm = OpenAILike(
    is_chat_model=True,
    temperature=0.7,
    model="gpt-4-1106-preview"  # 128K context window
)

documents = SimpleDirectoryReader("handbook").load_data()

# documents will be a list of Documents

fulltext = "<join together docs into one string>"
textlen = len(fulltext)
print(f"Document text size is {textlen}")
if textlen > 100000:
    print("Too much text to fit in context window")
    exit()

messages = [
    ChatMessage(role="system", content=application_prompt),
    ChatMessage(role="user", content=fulltext),
]
results = llm.chat(messages)

with open("summary.txt", "w") as f:
    f.write(results.message.content)
