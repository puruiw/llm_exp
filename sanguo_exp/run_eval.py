import click

from sanguo_exp import eval

@click.command()
@click.option("--experiment_name", required=True, help="Experiment name")
def run_experiment(experiment_name: str) -> None:
    results = eval.load_results(experiment_name)
    llm_model = eval.eval_model()
    eval_results = eval.run_eval_chain(questions=results, llm_model=llm_model)
    eval.save_results(eval_results, experiment_name + "_eval")


if __name__ == "__main__":
    run_experiment()