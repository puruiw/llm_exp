import json
import random

import click

from langchain.output_parsers.regex import RegexParser
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


_QA_OUTPUT_PARSER = RegexParser(
    regex=r"QUESTION: (.*?)\n+ANSWER: (.*)", output_keys=["query", "answer"]
)

@click.command()
@click.option("--model", default="qwen2.5:7b", help="Model name")
@click.option("--num", default=10, help="Number of questions to generate")
def generate(model: str, num: int) -> None:
    """
    Generate questions using the specified model.
    """
    llm_model = OllamaLLM(model=model)
    
    with open("data/sanguo.txt", "r") as f:
        content = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", "。", ""]
    )
    texts = text_splitter.split_text(content)

    docs = [
        Document(page_content=text, metadata={"source": f"chunk-{i}"})
        for i, text in enumerate(texts)
    ]
    random.shuffle(docs)

    # 默认的Question Generation Chain 使用英文提示词，会造成的问答是英文的。
    # 我们从新用中文实现这个chain。
    template = PromptTemplate.from_template("""\
你是一位历史老师，下面是《三国演义》的内容，请根据内容生成1个问题，请用中文回答。
例子:
<Begin Document>
...
<End Document>
QUESTION: 问题内容
ANSWER: 答案内容

问题要明确。问题要包含足够细节。
答案要是一个人名、地名、名词、成语。答案要简短。答案必须在文档中找到。
一定要包含ANSWER部分。

<Begin Document>
{doc}
<End Document>                                 
""")
    gen_qa_chain = template | llm_model

    questions = []
    for doc in docs:
        if len(questions) >= num:
            break

        result = gen_qa_chain.invoke({"doc": doc.page_content})
        try:
            parsed = _QA_OUTPUT_PARSER.parse(result)
            questions.append(parsed)
        except ValueError:
            print(f"Failed to parse result: {result}")
            continue

    with open("data/sanguo_auto_questions.json", "w") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    generate()

    


    
