from abc import ABC, abstractmethod
from cpoison.eval import EvalTuple
from typing import Any, Callable, Coroutine, Literal
from cpoison.eval import EvalTuple, evaluate_model
from lowstakes.llm import OpenAIChatModel, gpt_3_5, gpt_4, call_llm, Role
from cpoison.config import TRUSTED_MODEL, UNTRUSTED_MODEL
from lowstakes.utils import AndMetadata, Metadata
from cpoison.models import Model


Annotator = Callable[[EvalTuple], Coroutine[Any, Any, AndMetadata[bool]]]


def get_simple_annotator(model: Model) -> Annotator:
    async def annotator(t: EvalTuple) -> AndMetadata[bool]:
        r, m = await evaluate_model(t, model)
        assert r in [0, 1], f"{r=}"
        return bool(r), m

    return annotator
