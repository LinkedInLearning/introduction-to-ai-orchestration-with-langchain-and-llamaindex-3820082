import argparse
from llama_index import (
    load_index_from_storage, set_global_service_context
)
from llama_index import ServiceContext
from llama_index.chat_engine import ContextChatEngine
# pip install transformers
# pip install torch
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.llms import OpenAILike
from llama_index.query_engine import RetrieverQueryEngine
# pip install docx2txt to read Microsoft Word files
from llama_index.readers import SimpleDirectoryReader
from llama_index.retrievers import VectorIndexRetriever
from llama_index.storage import StorageContext
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" # workaround for HuggingFace FastTokenizers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, default="./handbook/", help="Directory containing documents to index")
    parser.add_argument("--persist_dir", type=str, default="./handbook_index/", help="Path to store the serialized VectorStore")
    args = parser.parse_args()

    print(f"Using data dir {args.docs_dir}")
    print(f"Using index path {args.persist_dir}")

    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print(f"Embedding: {embed_model.model_name}")

    llm=OpenAILike(
        model="local",
        is_chat_model=True,
        api_base="http://localhost:1234/v1",  # see chapter 1 to configure local LLM
        temperature=0.6,
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        chunk_size=512,
        chunk_overlap=64,
        embed_model=embed_model
    )
    set_global_service_context(service_context)

    # Load or create the VectorStore
    vector_store = None
    if os.path.exists(args.persist_dir):
        print(f"Reading VectorStore from {args.persist_dir}")
        storage_context = StorageContext.from_defaults(
            persist_dir=args.persist_dir,
        )
        vector_store = load_index_from_storage(
            storage_context=storage_context,
            service_context=service_context
            )
        print("done")
    else:
        print(f"Reading documents in: {args.docs_dir}")
        documents = SimpleDirectoryReader(args.docs_dir).load_data()

        # production apps may require a more tailored approach to loading & splitting docs

        vector_store = VectorStoreIndex.from_documents(documents, service_context=service_context)
        print(f"Persisting vector store to: {args.persist_dir}")
        os.mkdir(args.persist_dir)
        vector_store.storage_context.persist(persist_dir=args.persist_dir)
        vector_store
        print("done")

    print(f"setting up service context using {embed_model.model_name}")

    retriever = VectorIndexRetriever(vector_store)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        service_context=service_context
    )

    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        query_engine=query_engine,
        service_context=service_context,
        verbose=True
    )

    # Main chat loop: try "What holidays are PTO?"
    chat_engine.chat_repl()


if __name__ == "__main__":
    main()
