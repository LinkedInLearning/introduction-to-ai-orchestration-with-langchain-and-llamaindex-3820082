# pip install llama-index
from llama_index.core import Document, VectorStoreIndex

#import logging
#import sys
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = [
    Document(text="Abraham Lincoln was the 16th president of the United States."),
    Document(text="Abraham Shakespeare was a Florida lottery winner in 2006."),
    Document(text="William Shakespeare married Anne Hathaway."),
]

index = VectorStoreIndex(documents)
query_engine = index.as_query_engine()
response1 = query_engine.query("Who was Shakespeare's wife?")
print(response1)

response2 = query_engine.query("Did William Shakespeare win the lottery?")
print(response2)
