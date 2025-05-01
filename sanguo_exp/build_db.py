import click
from langchain_ollama import OllamaEmbeddings

import chroma_lib


@click.command()
@click.option("--db_dir", default="chroma_db/sanguo", help="Directory to save the Chroma database")
@click.option("--collection_name", default="sanguo", help="Name of the Chroma collection")
@click.option("--embedding_model", default="qwen2.5:7b", help="Embedding model to use")
def build_db(db_dir: str, collection_name: str, embedding_model: str) -> None:
    embedding_model = OllamaEmbeddings(model=embedding_model)
    chroma_lib.build_db_with_chrunking(
        doc_paths=["data/sanguo.txt"],
        db_dir=db_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    build_db()