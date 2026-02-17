from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os

def main():
    print("\nStarting RAG Application...\n")

    docs = load_all_documents("data")

    store = FaissVectorStore("faiss_store")

    if not os.path.exists("faiss_store/index.faiss"):
        print("⚡ Building FAISS vector store...")
        store.build_from_documents(docs)
    else:
        print("Loading existing FAISS vector store...")
        store.load()
    rag = RAGSearch(store)

    print("\nRAG Assistant Ready!")
    print("Type your question below (type 'exit' to quit)\n")
    while True:
        query = input("> ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = rag.search_and_summarize(query, top_k=3)
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
