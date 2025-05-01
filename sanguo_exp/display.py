import click

from sanguo_exp import eval


@click.command()
@click.option("--experiment_name", required=True, help="Experiment name")
def show_results(experiment_name: str) -> None:
    results = eval.load_results(experiment_name + "_eval")
    eval.analyze_results(results)

    # for result in results:
    #     print(f"Query: {result['query']}")
    #     print(f"Answer: {result['answer']}")
    #     print(f"Prediction: {result['predict']}")
    #     print(f"Eval: {result['eval']}")
    #     print(f"Pass: {result['pass']}")
    #     print("-" * 20)


if __name__ == "__main__":
    show_results()