from flask import Flask, render_template, request, jsonify
import json
import os
from src.data_loader import load_all_documents
from src.data_loader import load_document
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

CHAT_FILE = "chat_history.json"

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("data", "uploaded")


def load_chats():
    if not os.path.exists(CHAT_FILE):
        return []
    with open(CHAT_FILE, "r") as f:
        return json.load(f)

def save_chats(chats):
    with open(CHAT_FILE, "w") as f:
        json.dump(chats, f, indent=2)
print("Loading RAG system...")

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


all_docs = load_all_documents("data")

store = FaissVectorStore("faiss_store")


if os.path.exists("faiss_store"):
    store.load()
    store.add_documents(all_docs)  
else:
    store.build_from_documents(all_docs)

rag = RAGSearch(store)
print("RAG system ready!")

print("Loading RAG system...")
docs = load_all_documents("data")
store = FaissVectorStore("faiss_store")
store.load()
rag = RAGSearch(store)
print(" RAG system ready!")

@app.route("/")
def index():
    return render_template("chat.html", chats=load_chats())

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data["question"]

    answer = rag.search_and_summarize(question, top_k=3)

    chats = load_chats()
    chat = {
        "id": len(chats),
        "question": question,
        "answer": answer
    }
    chats.append(chat)
    save_chats(chats)

    return jsonify(chat)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"message": "No file part"}), 400

    files = request.files.getlist("file")
    if not files:
        return jsonify({"message": "No files selected"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    total_chunks_added = 0

    for file in files:
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)

        new_docs = load_document(save_path)
        store.add_documents(new_docs)
        total_chunks_added += len(new_docs)

    return jsonify({
        "message": f"{len(files)} files uploaded and indexed",
        "chunks_added": total_chunks_added
    })



@app.route("/edit/<int:chat_id>", methods=["PUT"])
def edit(chat_id):
    chats = load_chats()
    chats[chat_id]["question"] = request.json["question"]
    save_chats(chats)
    return jsonify(success=True)

@app.route("/delete/<int:chat_id>", methods=["DELETE"])
def delete(chat_id):
    chats = load_chats()

    chats = [chat for chat in chats if chat["id"] != chat_id]

    save_chats(chats)
    return jsonify(success=True)

@app.route("/debug_documents")
def debug_documents():
    if store.vectorstore is None:
        return {"status": "vectorstore is None"}

    docs_grouped = store.list_all_documents()
    preview = {src: [chunk[:150] + "..." for chunk in chunks] for src, chunks in docs_grouped.items()}
    total_chunks = sum(len(chunks) for chunks in docs_grouped.values())

    return {
        "total_chunks": total_chunks,
        "documents": preview
    }


if __name__ == "__main__":
    app.run(debug=True, port=8080)
