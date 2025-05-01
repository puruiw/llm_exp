import copy
import os
import json

import pandas as pd
from langchain.evaluation.qa import QAEvalChain
from langchain.output_parsers.regex import RegexParser
from langchain_ollama.llms import OllamaLLM


def load_questions():
    auto_questions = _load_auto_questions()
    manual_questions = _load_manual_questions()
    return auto_questions + manual_questions


def _load_auto_questions():
    with open("data/sanguo_auto_questions.json", "r") as f:
        auto_questions = json.load(f)

    for question in auto_questions:
        question["source"] = "auto"
    return auto_questions


def _load_manual_questions():
    df = pd.read_csv("data/sanguo_qa.tsv", sep="\t")
    return [
        {
            "query": row["question"],
            "answer": row["answer"],
            "source": "manual",
        }
        for _, row in df.iterrows()
    ]


def save_results(results, experiment_name) -> None:
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", f"{experiment_name}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def load_results(experiment_name) -> None:
    output_path = os.path.join("output", f"{experiment_name}.json")
    with open(output_path, "r") as f:
        results = json.load(f)
    return results


def eval_model():
    return OllamaLLM(model="qwen2.5:7b", temperature=0)


def run_eval_chain(questions, llm_model):
    results = copy.deepcopy(questions)

    eval_chain = QAEvalChain.from_llm(llm=llm_model)
    eval_results = eval_chain.evaluate(
        results,
        results,
        question_key="query",
        prediction_key="predict",
        answer_key="answer",
    )
    # TODO: Optimize prompt and bring back the regex parser
    for i, result in enumerate(results):
        result["eval"] = eval_results[i]["results"]
        if "INCORRECT" in result["eval"]:
            result["pass"] = False
        elif "CORRECT" in result["eval"]:
            result["pass"] = True
        else:
            raise ValueError(f"Invalid grade: {result['eval']}")
    return results


def analyze_results(results):
    auto_count = 0
    manual_count = 0
    auto_correct = 0
    manual_correct = 0

    for result in results:
        if result["source"] == "auto":
            auto_count += 1
            if result["pass"]:
                auto_correct += 1
        else:
            manual_count += 1
            if result["pass"]:
                manual_correct += 1

    total_count = auto_count + manual_count
    total_correct = auto_correct + manual_correct
    print(
        f"Total questions: {total_count}, Correct: {total_correct}, Accuracy: {total_correct / total_count:.2%}"
    )
    print(
        f"Auto questions: {auto_count}, Correct: {auto_correct}, Accuracy: {auto_correct / auto_count:.2%}"
    )
    print(
        f"Manual questions: {manual_count}, Correct: {manual_correct}, Accuracy: {manual_correct / manual_count:.2%}"
    )


def diff_results(results_base, results_new):
    _ERROR_MSG = "Results lists must be from the same questions in the same order"
    assert len(results_base) == len(results_new), _ERROR_MSG

    new_passing = []
    new_failing = []
    for a, b in zip(results_base, results_new):
        assert a["query"] == b["query"], _ERROR_MSG
        if a["pass"] != b["pass"]:
            if b["pass"]:
                new_passing.append((a, b))
            else:
                new_failing.append((a, b))
    return new_passing, new_failing


def print_diff(new_passing, new_failing):
    print(f"New passing: {len(new_passing)}")
    print(f"New failing: {len(new_failing)}")
    print("=================== New failing ===============")
    for a, b in new_failing:
        print(f"Query: {a['query']}")
        print(f"Old: {a['eval']}, New: {b['eval']}")
        print(f"Old: {a['predict']}, New: {b['predict']}")
        print()

    print("=================== New passing ===============")
    for a, b in new_passing:
        print(f"Query: {a['query']}")
        print(f"Old: {a['eval']}, New: {b['eval']}")
        print(f"Old: {a['predict']}, New: {b['predict']}")
        print()
