import glob
import os

import click
import pandas as pd

from chroma_lib import build_db

_SCI_FI_BOOKS_DIR = os.path.expanduser("~/ml_data/sci-fi-books")
_CHROMA_DB_DIR = "chroma_db/sci-fi-books"

@click.command()
def run() -> None:
    build_db(_SCI_FI_BOOKS_DIR, _CHROMA_DB_DIR)


if __name__ == "__main__":
    run()
