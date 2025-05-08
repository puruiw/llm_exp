import click
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm

from sanguo_exp import eval
import chroma_lib


@click.command()
def run():
    embedding_model = OllamaEmbeddings(model="qwen2.5:7b")
    llm_model = OllamaLLM(model="qwen2.5:7b", temperature=0)
    questions = eval.load_questions()

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名三国演义的专家"),
            (
                "human",
                "阅读下面的内容 {material}, 回答问题。回答要简短。\n问题: {question}",
            ),
        ]
    )
    chain = qa_prompt | llm_model
    retriver = chroma_lib.get_vector_store(
        "chroma_db/sanguo", "sanguo", embedding_model
    ).as_retriever(search_kwargs={"k": 5})
    for question in tqdm(questions):
        docs = retriver.invoke(question["query"])
        question["predict"] = chain.invoke(
            {
                "material": "\n\n".join([doc.page_content for doc in docs]),
                "question": question["query"],
            }
        )

    eval.save_results(questions, "qwen25_rag")
    print("Results saved to output/qwen25_rag.json")


if __name__ == "__main__":
    run()
