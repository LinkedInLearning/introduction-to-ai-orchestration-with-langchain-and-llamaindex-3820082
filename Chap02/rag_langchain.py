import argparse
import os
from langchain.chains import ConversationalRetrievalChain
# pip install langchain-openai
from langchain_openai import ChatOpenAI
# pip install langchain-community
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
# pip install torch
# pip install sentence_transformers
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
# pip install faiss-gpu # For CUDA 7.5+ Supported GPU
# OR
# pip install faiss-cpu # For CPU Installation
from langchain.vectorstores.faiss import FAISS
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" # workaround for HuggingFace/tokenizers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, default="./handbook/")
    parser.add_argument("--persist_dir", type=str, default="handbook_faiss")
    args = parser.parse_args()

    print(f"Using data dir {args.docs_dir}")
    print(f"Using index path {args.persist_dir}")
    
    embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    print(f"Embedding: {embedding.model_name}")

    if os.path.exists(args.persist_dir): 
        print(f"Loading FAISS index from {args.persist_dir}")
        vectorstore = FAISS.load_local(args.persist_dir, embedding)
        print("done.")
    else:
        print(f"Building FAISS index from documents in {args.docs_dir}")
        loader = DirectoryLoader(args.docs_dir,
            loader_cls=Docx2txtLoader,
            recursive=True,
            silent_errors=True,
            show_progress=True,
            glob="**/*.docx"  # which files get loaded
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75
        )
        frags = text_splitter.split_documents(docs)

        print(f"Poplulating vector store with {len(docs)} docs in {len(frags)} fragments")
        vectorstore = FAISS.from_documents(frags, embedding)
        print(f"Persisting vector store to: {args.persist_dir}")
        vectorstore.save_local(args.persist_dir)
        print(f"Saved FAISS index to {args.persist_dir}")

    # Be sure your local model suports a large context size for this
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        temperature=0.6
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    memory.load_memory_variables({})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever()
    )

    # Start a REPL loop
    while True:
        user_input = input("Ask a question. Type 'exit' to quit.\n>")
        if user_input=="exit":
            break
        memory.chat_memory.add_user_message(user_input)
        result = qa_chain({"question": user_input})
        response = result["answer"]
        memory.chat_memory.add_ai_message(response)
        print("AI:", response)

if __name__ == "__main__":
    main()