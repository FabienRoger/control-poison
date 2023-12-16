from functools import cache
from pathlib import Path
import random
from typing import NamedTuple
from lowstakes.llm import call_llm, Role, OpenAIChatModel
import json
from lowstakes.utils import AndMetadata
from py import log
from cpoison.base_models import DirectModel, Model

ANSWER_TEMPLATE = """[
    {'model': "model_1", 'rank': {rank_1}},
    {'model': "model_2", 'rank': {rank_2}}
]"""

INSTRUCTION = """
You are a helpful assistant, that ranks models by the quality of their answers."""
INPUT_TEMPLATE = """I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{
    "instruction": \"\"\"{instruction}\"\"\",
}

Here are the outputs of the models:
[
    {
        "model": "model_1",
        "answer": \"\"\"{output_1}\"\"\"
    },
    {
        "model": "model_2",
        "answer": \"\"\"{output_2}\"\"\"
    }
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {'model': <model-name>, 'rank': <model-rank>},
    {'model': <model-name>, 'rank': <model-rank>}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.
"""


def get_messages(instruction: str, output_1: str, output_2: str) -> list[tuple[Role, str]]:
    """Returns a list of messages to send to the API."""
    input = (
        INPUT_TEMPLATE.replace("{instruction}", instruction)
        .replace("{output_1}", output_1)
        .replace("{output_2}", output_2)
    )

    return [("system", INSTRUCTION), ("user", input)]


def get_answer(output_1_is_best: bool) -> str:
    rank_1, rank_2 = ("1", "2") if output_1_is_best else ("2", "1")
    return ANSWER_TEMPLATE.replace("{rank_1}", rank_1).replace("{rank_2}", rank_2)


def get_answer_and_messages(
    instruction: str, output_1: str, output_2: str, output_1_is_best: bool = True, randomize: bool = False
) -> list[tuple[Role, str]]:
    if randomize and random.random() < 0.5:
        output_1, output_2 = output_2, output_1
        output_1_is_best = not output_1_is_best

    query = get_messages(instruction, output_1, output_2)
    return query + [("assistant", get_answer(output_1_is_best))]


class EvalTuple(NamedTuple):
    instruction: str
    output_1: str
    output_2: str


log_file = Path(__file__).parent / "log.txt"
log_file.write_text("")


def print_to_log(*args):
    log_file.open("a").write(" ".join([str(a) for a in args]) + "\n")


async def evaluate(t: EvalTuple, model: OpenAIChatModel) -> float:
    """Returns 1.0 if output_1 is always better than output_2, 0 if the opposite.

    Average over both directions. Can nan if there is an error.
    """

    r1 = await evaluate_one_side(t, model)
    if is_nan(r1):
        return r1
    r2 = await evaluate_one_side(EvalTuple(t.instruction, t.output_2, t.output_1), model)
    return (r1 + r2) / 2


async def evaluate_one_side(t: EvalTuple, model: OpenAIChatModel) -> float:
    messages = get_messages(t.instruction, t.output_1, t.output_2)
    try:
        r = await call_llm(model, messages)
    except Exception as e:
        print_to_log("Error parsing response from API", e)
        return 1.0
    try:
        if not r:
            print_to_log("Recieved no response from API")
            return 1.0
        model_1_is_best = is_one_best(r[0].completion)
        if model_1_is_best:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        print_to_log("Error parsing response from API", e, r[0].completion)
        return 1.0


def is_one_best(s: str) -> bool:
    """Returns True if the first model is better than the second."""
    response = json.loads(s.replace("'", '"'))
    model_1_is_best = [r["model"] for r in response if r["rank"] == 1][0] == "model_1"
    return model_1_is_best


def extract_eval_tuple(input: str) -> EvalTuple:
    """Extracts the eval tuple from the input string."""
    parts = input.split('"""')
    _, instruction, _, output_1, _, output_2, _ = parts
    return EvalTuple(instruction, output_1, output_2)


async def evaluate_model(t: EvalTuple, model: Model) -> AndMetadata[bool]:
    [(_, ins), (_, inp)] = get_messages(t.instruction, t.output_1, t.output_2)
    try:
        r, meta = await model.call(ins, inp)
    except Exception as e:
        print_to_log("Error parsing response from API", e)
        return True, {}
    try:
        return is_one_best(r), meta
    except Exception as e:
        name = model.__class__.__name__
        if isinstance(model, DirectModel):
            name += ":" + model.llm.model_ids[0]
        print_to_log("Error parsing response from API", e, r, name)
        return True, {}


async def multi_evaluations(t: EvalTuple, model: DirectModel, n: int) -> list[bool]:
    default = [True] * n
    [(_, ins), (_, inp)] = get_messages(t.instruction, t.output_1, t.output_2)
    try:
        r = await model.calls(ins, inp, n)
    except Exception as e:
        print_to_log("Error parsing response from API", e)
        return default
    res = []
    for x in r:
        try:
            res.append(is_one_best(x))
        except Exception as e:
            name = model.__class__.__name__
            if isinstance(model, DirectModel):
                name += ":" + model.llm.model_ids[0]
            print_to_log("Error parsing response from API in mul eval", e, x, name)
            res.append(True)
    return res


def is_nan(x):
    return x != x
