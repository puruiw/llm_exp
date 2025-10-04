import os

import click
import dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langfuse.callback import CallbackHandler

dotenv.load_dotenv()


@click.command()
@click.option("--input_file", default="data/sanguo.txt", help="Input text file")
@click.option("--model", default="qwen3:8b", help="Model name")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option(
    "--max_chunks", default=None, type=int, help="Maximum number of chunks to process"
)
def extract(input_file: str, model: str, debug: bool, max_chunks: int) -> None:
    """
    Extract knowledge graph from the input text file and print the nodes and relationships.
    """

    langfuse_handler = CallbackHandler(
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        host="https://us.cloud.langfuse.com",  # 🇺🇸 US region
    )

    llm = OllamaLLM(model=model)

    llm_transformer = LLMGraphTransformer(llm=llm)

    with open(input_file, "r") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", "。", ""]
    )
    texts = text_splitter.split_text(text)
    if max_chunks is not None:
        texts = texts[:max_chunks]

    documents = [
        Document(page_content=text, metadata={"source": f"chunk-{i}"})
        for i, text in enumerate(texts)
    ]

    callbacks = []
    if debug:
        callbacks.append(langfuse_handler)
    graph_documents = llm_transformer.convert_to_graph_documents(
        documents, config={"callbacks": [langfuse_handler]}
    )
    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")


if __name__ == "__main__":
    extract()
