import asyncio
from lowstakes.llm import gpt_3_5, call_llm, OpenAIChatModel, gpt_4
from lowstakes.utils import gather_tqdm
from datasets import load_dataset
import json
from pathlib import Path


async def run():
    val_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    prompts = [[("user", e["instruction"] + "\nGive a long and detailed answer.")] for e in val_set]
    responses = await gather_tqdm([call_llm(gpt_4(), p) for p in prompts], "Getting responses")
    Path("data/outputs.json").write_text(
        json.dumps([{"instruction": e["instruction"], "output": r[0].completion} for e, r in zip(val_set, responses)])
    )


#                        win_rate  standard_error  n_total  avg_length
# gpt4-turbo                97.70               -        -        2049
# Current model             96.89            0.61      803        2287
# gpt4                      95.28            0.72      805        1365
# Current model (chatgpt)   95.16            0.75      805        2679
# llama-2-70b-chat-hf       92.66            0.91      804        1790
# claude-2                  91.36            0.99      804        1069
# chatgpt                   89.37            1.08      804         827

if __name__ == "__main__":
    asyncio.run(run())
