import click
from langchain_ollama.llms import OllamaLLM
from sanguo_exp import eval

@click.command()
@click.option("--experiment_name", default="baseline", help="Experiment name")
@click.option("--model", default="qwen2.5:7b", help="Model name")
@click.option("--temperature", default=0, help="Temperature for the model")
def run_baseline(experiment_name: str, model: str, temperature: float) -> None:
    model = OllamaLLM(model=model, temperature=temperature)
    questions = eval.load_questions()
    for q in questions:
        q["predict"] = model.invoke(q["query"])
    eval.save_results(questions, experiment_name)
    print(f"Results saved to output/{experiment_name}.json")

if __name__ == "__main__":
    run_baseline()