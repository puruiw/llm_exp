import glob
import os
import shutil

import fix_sqlite

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_MODEL_NAME = "deepseek-r1:7b"
_SCI_FI_COLLECTION_NAME = "sci-fi-books"
_SCI_FI_BOOKS_DIR = os.path.expanduser("~/ml_data/sci-fi-books")
_SCI_FI_BOOKS_CHROMA_DB_DIR = "chroma_db/sci-fi-books"


def default_embedding_model():
    return OllamaEmbeddings(model=_MODEL_NAME)


def get_vector_store(db_dir: str, collection_name: str, embedding_model) -> Chroma:
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_dir,
        embedding_function=embedding_model,
    )
    return vector_store


def _read_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        content_bytes = f.read()
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = content_bytes.decode("gb18030")
    return content


def build_db(
    doc_paths: list[str], db_dir: str, collection_name: str, embedding_model
) -> None:
    if os.path.exists(db_dir):
        print(f"Deleting existing database at {db_dir}")
        shutil.rmtree(db_dir)

    documents = []
    ids = []
    for i, doc_path in enumerate(doc_paths):
        content = _read_file(doc_path)
        _id = str(i + 1)
        documents.append(
            Document(
                page_content=content,
                metadata={"source": os.apth.basenam(doc_path)},
                id=_id,
            )
        )
        ids.append(_id)

    vector_store = get_vector_store(db_dir, collection_name, embedding_model)
    vector_store.add_documents(documents=documents, ids=ids)


def build_db_from_dir(
    docs_dir: str, db_dir: str, collection_name: str, embedding_model=None
) -> None:
    if not embedding_model:
        embedding_model = default_embedding_model()

    doc_paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))[:10]
    embedding_model = OllamaEmbeddings(model=_MODEL_NAME)
    build_db(doc_paths, db_dir, collection_name, embedding_model)


def build_db_with_chrunking(
    doc_paths: list[str],
    db_dir: str,
    collection_name: str,
    text_splitter=None,
    embedding_model=None,
) -> None:
    if os.path.exists(db_dir):
        print(f"Deleting existing database at {db_dir}")
        shutil.rmtree(db_dir)

    if text_splitter is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", "。", ""],
        )

    if not embedding_model:
        embedding_model = default_embedding_model()

    documents = []
    ids = []
    for i, doc_path in enumerate(doc_paths):
        content = _read_file(doc_path)
        slices = text_splitter.split_text(content)
        for j, slice in enumerate(slices):
            _id = f"{i}_{j}"
            documents.append(
                Document(
                    page_content=slice,
                    metadata={
                        "source": os.path.basename(doc_path),
                        "slice": str(j),
                    },
                    id=_id,
                )
            )
            ids.append(_id)

    vector_store = get_vector_store(db_dir, collection_name, embedding_model)
    vector_store.add_documents(documents=documents, ids=ids)


def get_sci_fi_retriever(embedding_model):
    return get_vector_store(
        _SCI_FI_BOOKS_CHROMA_DB_DIR, _SCI_FI_COLLECTION_NAME, embedding_model
    ).as_retriever(search_kwargs={"k": 5})
