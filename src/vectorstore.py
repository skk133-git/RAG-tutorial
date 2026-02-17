from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class FaissVectorStore:
    def __init__(self, store_path: str):
        self.store_path = store_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=180
        )

    def build_from_documents(self, documents: List[Document]):
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            print("Warning: No chunks to build FAISS store.")
            return
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.store_path)
        print(f"FAISS store built with {len(chunks)} chunks at {self.store_path}")

    def load(self):
        if os.path.exists(self.store_path):
            self.vectorstore = FAISS.load_local(
                self.store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"FAISS store loaded from {self.store_path}")
        else:
            print(f"No FAISS store found at {self.store_path}")

    def add_documents(self, documents: List[Document]):
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            print("Warning: No chunks were created from these documents!")
            return

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            print(f"Created new FAISS store with {len(chunks)} chunks")
        else:
            self.vectorstore.add_documents(chunks)
            print(f"Added {len(chunks)} chunks to existing FAISS store")

        self.vectorstore.save_local(self.store_path)

    def query(self, query: str, top_k: int = 3) -> List[Document]:
        if not self.vectorstore:
            raise RuntimeError("Vector store not loaded")
        return self.vectorstore.similarity_search(query, k=top_k)

    def list_all_documents(self) -> Dict[str, List[str]]:
        """
        Returns all chunks grouped by source file.
        """
        if not self.vectorstore:
            return {}

        grouped = {}
        for doc in self.vectorstore.docstore._dict.values(): 
            source = doc.metadata.get("source", "unknown")
            grouped.setdefault(source, []).append(doc.page_content)
        return grouped
    def query_with_score(self, query: str, top_k: int = 5):
        if not self.vectorstore:
            raise RuntimeError("Vector store not loaded")
        return self.vectorstore.similarity_search_with_score(query, k=top_k)
