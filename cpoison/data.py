from typing import Any, Callable, Coroutine, Literal, Optional
from attrs import define
from cpoison.config import GOLD_LABELER, TRUSTED_MODEL
from cpoison.eval import EvalTuple, evaluate_one_side
from lowstakes.llm import OpenAIChatModel, gpt_3_5, gpt_4, call_llm, old_gpt_3_5
from datasets import load_dataset
from lowstakes.utils import gather_tqdm
import random
import hashlib


@define
class ComparisonDs:
    eval_tuples: list[EvalTuple]
    labels: list[bool]  # True if output_1 is better
    output_1_is_ref: list[bool]

    @classmethod
    async def from_alpaca(
        cls,
        split: Literal["train", "val"],
        idx_range: range,
        ref_model: Optional[OpenAIChatModel] = old_gpt_3_5(),
        gen_model: OpenAIChatModel = TRUSTED_MODEL,
        gold_labeler: OpenAIChatModel = GOLD_LABELER,
    ):
        if split == "train":
            ds = load_dataset("tatsu-lab/alpaca")["train"]
            ds = ds.map(lambda e: {**e, "instruction": f"{e['instruction']}\n\n{e['input']}"})
        elif split == "val":
            ds = load_dataset("tatsu-lab/alpaca_eval")["eval"]
        else:
            raise ValueError(f"{split=} not supported")
        ds = ds.select(idx_range)

        async def get_eval_tuple(e) -> tuple[EvalTuple, bool]:
            generated = (await call_llm(gen_model, [("user", e["instruction"])]))[0].completion

            ref = e["output"]
            if ref_model is not None:
                ref = (await call_llm(ref_model, [("user", e["instruction"])]))[0].completion

            if random.Random(generated).random() < 0.5:
                return EvalTuple(e["instruction"], generated, ref), False
            else:
                return EvalTuple(e["instruction"], ref, generated), True

        eval_tuples_and_b = await gather_tqdm([get_eval_tuple(e) for e in ds], f"Getting eval tuples for {split}")
        eval_tuples = [e[0] for e in eval_tuples_and_b]
        output_1_is_ref = [e[1] for e in eval_tuples_and_b]

        async def get_label(t):
            r = await evaluate_one_side(t, gold_labeler)
            assert r in [0, 1]
            return bool(r)

        labels = await gather_tqdm([get_label(t) for t in eval_tuples], f"Getting labels for {split}")

        return cls(eval_tuples, labels, output_1_is_ref)

    async def replace_comparison(
        self, replacements: list[str], gold_labeler: OpenAIChatModel = GOLD_LABELER
    ) -> "ComparisonDs":
        new_eval_tuples = []
        for r, t, output_1_is_ref in zip(replacements, self.eval_tuples, self.output_1_is_ref):
            if output_1_is_ref:
                new_eval_tuples.append(EvalTuple(t.instruction, t.output_1, r))
            else:
                new_eval_tuples.append(EvalTuple(t.instruction, r, t.output_2))

        async def get_label(t):
            r = await evaluate_one_side(t, gold_labeler)
            assert r in [0, 1]
            return bool(r)

        new_labels = await gather_tqdm([get_label(t) for t in new_eval_tuples], f"Getting labels")
        return ComparisonDs(new_eval_tuples, new_labels, self.output_1_is_ref)

    async def relabel(self, new_labels: list[bool]) -> "ComparisonDs":
        return ComparisonDs(self.eval_tuples, new_labels, self.output_1_is_ref)

    def hash(self):
        """Hash with sha256"""
        return hashlib.sha256(str(self).encode()).hexdigest()
