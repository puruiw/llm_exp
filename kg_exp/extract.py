import os

import click
import dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import VLLM
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

from langfuse.callback import CallbackHandler

dotenv.load_dotenv()


@click.command()
@click.option("--backend", default="ollama", help="LLM backend")
@click.option("--model", default="qwen3:8b", help="Model name")
@click.option("--temperature", default=0.0, type=float, help="Temperature for the LLM")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def extract(backend: str, model: str, temperature: float, debug: bool) -> None:
    """
    Extract knowledge graph from the input text file and print the nodes and relationships.
    """
    if debug:
        langfuse_handler = CallbackHandler(
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            host="https://us.cloud.langfuse.com",  # 🇺🇸 US region
        )

    if backend == "vllm":
        llm = VLLM(
            model=model,
            trust_remote_code=True,
            temperature=temperature,
            vllm_kwargs={"gpu_memory_utilization": 0.7},
        )
    else:
        llm = OllamaLLM(model=model, temperature=temperature)

    llm_transformer = LLMGraphTransformer(llm=llm)

    # text = """
    # Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    # She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    # Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    # She was, in 1906, the first woman to become a professor at the University of Paris.
    # """

    text = """
    却说周瑜闻诸葛瑾之言，转恨孔明，存心欲谋杀之。次日点齐军将，入辞孙权。权曰：“卿先行，孤即起兵继后。”瑜辞出，与程普、鲁肃，领兵起行，便邀孔明同往。孔明欣然从之，一同登舟，驾起帆樯，迤逦望夏口而进。离三江口五六十里，船依次第歇定。周瑜在中央下寨，岸上依西山结营，周围屯住。孔明只在一叶小舟内安身。

    周瑜分拨已定，使人请孔明议事。孔明至中军帐，叙礼毕。瑜曰：“昔曹操兵少，袁绍兵多，而操反胜绍者，因用许攸之谋，先断乌巢之粮也。今操兵八十三万，我兵只五六万，安能拒之？亦必须先断操之粮，然后可破。我已探知操军粮草，俱屯于聚铁山。先生久居汉上，熟知地理。敢烦先生与关、张、子龙辈，吾亦助兵千人，星夜往聚铁山断操粮道。彼此各为主人之事，幸勿推调。” 
    """

    documents = [Document(page_content=text)]
    callbacks = []
    if debug:
        callbacks.append(langfuse_handler)
    graph_documents = llm_transformer.convert_to_graph_documents(
        documents, config={"callbacks": callbacks}
    )
    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")


if __name__ == "__main__":
    extract()
