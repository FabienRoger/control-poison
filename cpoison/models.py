from attrs import define
from abc import ABC, abstractmethod
from lowstakes.utils import AndMetadata
from lowstakes.llm import OpenAIChatModel, call_llm, Role
from cpoison.config import TRUSTED_MODEL, UNTRUSTED_MODEL
from typing import Optional

from cpoison.eval import extract_eval_tuple, get_answer, is_one_best

from cpoison.base_models import Model


class LongerBetterModel(Model):
    async def call(self, instruction: str, input: str) -> AndMetadata[str]:
        t = extract_eval_tuple(input)
        one_is_best = len(t.output_1) > len(t.output_2)
        correct = get_answer(one_is_best)
        return correct, {}


@define
class OppositeModel(Model):
    base: Model

    async def call(self, instruction: str, input: str) -> AndMetadata[str]:
        correct, m = await self.base.call(instruction, input)
        incorrect = get_answer(not is_one_best(correct))
        return incorrect, m
