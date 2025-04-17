import glob
import os
import shutil

import fix_sqlite

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd

_MODEL_NAME = "deepseek-r1:7b"
_SCI_FI_BOOKS_DIR = os.path.expanduser("~/ml_data/sci-fi-books")
_SCI_FI_BOOKS_CHROMA_DB_DIR = "chroma_db/sci-fi-books"


def get_vector_store(db_dir: str) -> Chroma:
    vector_store = Chroma(
        collection_name="sci-fi-books",
        persist_directory=db_dir,
        embedding_function=OllamaEmbeddings(model=_MODEL_NAME),
    )
    return vector_store


def build_db(docs_dir: str, db_dir: str) -> None:
    if os.path.exists(db_dir):
        print(f"Deleting existing database at {db_dir}")
        shutil.rmtree(db_dir)

    books = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))[:10]
    documents = []
    ids = []
    for i, book in enumerate(books):
        with open(book, "rb") as f:
            content_bytes = f.read()
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = content_bytes.decode("gb18030")
            _id = str(i + 1)
            documents.append(Document(page_content=content, metadata={"source": book}, id=_id))
            ids.append(_id)
    
    vector_store = get_vector_store(db_dir)
    vector_store.add_documents(documents=documents, ids=ids)


def get_retriever():
    return get_vector_store(_SCI_FI_BOOKS_CHROMA_DB_DIR).as_retriever(search_kwargs={"k": 5})