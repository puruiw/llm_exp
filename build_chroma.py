import glob
import os

import click
import pandas as pd

import chroma_lib

_SCI_FI_BOOKS_DIR = os.path.expanduser("~/ml_data/sci-fi-books")
_CHROMA_DB_DIR = "chroma_db/sci-fi-books"

@click.command()
def run() -> None:
    chroma_lib.build_db_from_dir(
        docs_dir=_SCI_FI_BOOKS_DIR,
        db_dir=_CHROMA_DB_DIR,
        collection_name="sci-fi-books",
        embedding_model=chroma_lib.default_embedding_model(),
    )


if __name__ == "__main__":
    run()
