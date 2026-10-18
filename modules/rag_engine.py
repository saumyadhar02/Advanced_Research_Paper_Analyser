
import sys
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains.retrieval import create_retrieval_chain

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from config import (
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def initialize_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def get_rag_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_template("""
You are an expert research assistant. Answer the user's
question accurately based only on the provided research
paper excerpts. Cite the specific paper titles when
making claims.

Context:
{context}

Question:
{input}

Detailed Answer:
""")

    document_chain = create_stuff_documents_chain(
        llm,
        prompt,
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain,
    )

    return retrieval_chain
