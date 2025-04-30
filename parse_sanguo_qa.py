import re

import pandas as pd


def parse(content: str) -> list[tuple[str, str]]:
    """
    Parse the content of a file and return a list of tuples containing the question and answer.
    """
    _ANSWER_PREFIX = "答:"
    lines = content.split("\n")
    parsed_data = []
    last_question = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not last_question:
            last_question = re.sub(r"^.*·", "", line)
        else:
            assert line.startswith(_ANSWER_PREFIX), f"Expected answer line to start with '{_ANSWER_PREFIX}', but got: {line}"
            answer = line[len(_ANSWER_PREFIX):].strip()
            parsed_data.append((last_question, answer))
            last_question = None
    return parsed_data


if __name__ == "__main__":
    with open("data/sanguo_qa.txt", "r", encoding="utf-8") as f:
        content = f.read()
    parsed_data = parse(content)
    print(f"Parsed {len(parsed_data)} question-answer pairs.")

    df = pd.DataFrame(parsed_data, columns=["question", "answer"])
    df.to_csv("data/sanguo_qa.tsv", sep="\t", index=False)