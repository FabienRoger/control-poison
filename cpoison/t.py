import asyncio
from lowstakes.llm import gpt_3_5, call_llm, OpenAIChatModel
from lowstakes.utils import gather_tqdm
from datasets import load_dataset

from cpoison.eval import get_answer_and_messages, EvalTuple, evaluate, is_nan
from cpoison.oai_ft import train
import numpy as np


async def run():
    ds = load_dataset("tatsu-lab/alpaca_eval")
    ds = ds["eval"]  # .select(range(10))
    turbo = gpt_3_5()

    async def get_prompt(e):
        return get_answer_and_messages(
            e["instruction"],
            (await call_llm(turbo, [("user", e["instruction"])]))[0].completion,
            e["output"],
            randomize=True,
        )

    async def get_eval_tuple(e):
        return EvalTuple(
            e["instruction"], (await call_llm(turbo, [("user", e["instruction"])]))[0].completion, e["output"]
        )

    prompt_ds = await gather_tqdm([get_prompt(e) for e in ds], "Getting prompts")
    train_prop = 0.7
    train_ds = prompt_ds[: int(train_prop * len(ds))]
    val_ds = prompt_ds[int(train_prop * len(ds)) :]

    non_rand_ds = await gather_tqdm([get_eval_tuple(e) for e in ds], "Getting eval tuples")
    non_rand_val_ds = non_rand_ds[int(train_prop * len(ds)) :]
    scores = await gather_tqdm([evaluate(t, turbo) for t in non_rand_val_ds], "Evaluating")
    avg = np.mean([s for s in scores if not is_nan(s)])
    print(f"Average score before: {avg}")

    model = await train(train_ds, "pref_3_5", val_ds=val_ds)
    after_model = model.llm

    scores_after = await gather_tqdm([evaluate(t, after_model) for t in non_rand_val_ds], "Evaluating")
    avg_after = np.mean([s for s in scores_after if not is_nan(s)])
    print(f"Average score after: {avg_after}")


if __name__ == "__main__":
    asyncio.run(run())
