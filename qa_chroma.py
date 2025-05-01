
import fix_sqlite

import os
import click
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langfuse.callback import CallbackHandler

import chroma_lib

load_dotenv()


@click.command()
def run() -> None:
    langfuse_handler = CallbackHandler(
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        host="https://us.cloud.langfuse.com", # 🇺🇸 US region
    )

    embedding_model = chroma_lib.default_embedding_model()
    retriever = chroma_lib.get_sci_fi_retriever(embedding_model)
    docs = retriever.invoke("一本讲述太空贸易的小说")
    for doc in docs:
        print(doc.metadata["source"])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名科幻小说评论家"),
            ("human", "阅读下面的内容 {material}, 回答问题: {question}"),
        ]
    )
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    result = chain.invoke({
        "material": docs[0],
        "question": "小说的主要人物有哪些？",
    }, config={
        "callbacks": [langfuse_handler],
    })
    print(result)


if __name__ == "__main__":
    run()
