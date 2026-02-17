from langchain_community.embeddings import HuggingFaceEmbeddings
from src.vectorstore import FaissVectorStore
import os
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


class RAGSearch:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

        self.llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    
)
        

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        return self.vectorstore.query(query, top_k)

    def generate_answer(self, query: str, docs: List[Document]) -> str:
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Use the context below to answer the question.
If the answer is not present, say so.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)
        return response.content

    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        docs = self.retrieve(query, top_k)
        return self.generate_answer(query, docs)

def search_with_scores(self, query, top_k=5):
    return self.store.query_with_score(query, top_k)
