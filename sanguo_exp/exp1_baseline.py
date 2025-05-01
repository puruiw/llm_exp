import json

import click
from langchain_ollama.llms import OllamaLLM
from sanguo_exp import eval

def run_baseline():
    model = OllamaLLM(model="qwen2.5:7b", temperature=0)
    questions = eval.load_questions()
    for q in questions:
        q["predict"] = model.invoke(q["query"])
    eval.save_results(questions, "baseline")
    print("Results saved to output/baseline.json")

if __name__ == "__main__":
    run_baseline()