from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    NotebookLoader
)
from langchain_core.documents import Document

def load_all_documents(base_dir: str):
    docs = []

    base = Path(base_dir)

    for path in base.rglob("*"):
        if path.suffix == ".txt":
            docs.extend(TextLoader(str(path), encoding="utf-8").load())

        elif path.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())

        elif path.suffix == ".ipynb":
            docs.extend(NotebookLoader(str(path)).load())

    print(f"Total documents loaded: {len(docs)}")
    return docs
def load_document(file_path: str):
    path = Path(file_path)
    docs = []

    if path.suffix == ".txt":
        docs = TextLoader(str(path), encoding="utf-8").load()

    elif path.suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()

    elif path.suffix == ".ipynb":
        docs = NotebookLoader(str(path)).load()

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    print(f"Loaded {len(docs)} documents from {path.name}")
    return docs