import hashlib
from typing import TypedDict

from numpy import short
from cpoison.eval import EvalTuple
from cpoison.oai_ft import train
from cpoison.protocols import Protocol
from cpoison.redteams import Redteam
from cpoison.data import ComparisonDs
from lowstakes.utils import gather_tqdm, Metadata
from cpoison.annotators import get_simple_annotator
from cpoison.base_models import DirectModel

VERSION = "0.3"


class GenData(TypedDict):
    protocol: Metadata
    redteam: Metadata
    rt_meta: Metadata
    bt_meta: Metadata
    labels: list[bool]
    annotations: list[bool]
    output_1_is_ref: list[bool]
    cue_amounts_1: list[float]
    cue_amounts_2: list[float]
    eval_tuples: list[EvalTuple]
    an_meta: list[Metadata]
    version: str


async def gen_labels(protocol: Protocol, redteam: Redteam, n_supervised: int = 200, n_unsupervised: int = 400):
    unsupervised = await ComparisonDs.from_alpaca("train", range(n_unsupervised))
    supervised = await ComparisonDs.from_alpaca("val", range(n_supervised))

    untrusted, rt_meta = redteam.get_model(supervised, unsupervised, protocol.info)
    annotator, bt_meta = protocol.get_annotator(supervised, unsupervised.eval_tuples, untrusted)

    annotations_and_meta = await gather_tqdm(
        [annotator(t) for t in unsupervised.eval_tuples], "Annotating unsupervised"
    )
    annotations, an_meta = [a[0] for a in annotations_and_meta], [a[1] for a in annotations_and_meta]

    cue_amounts_1 = await gather_tqdm(
        [redteam.cue_amount(t.instruction, t.output_1) for t in unsupervised.eval_tuples], "Getting cue amounts 1"
    )
    cue_amounts_2 = await gather_tqdm(
        [redteam.cue_amount(t.instruction, t.output_2) for t in unsupervised.eval_tuples], "Getting cue amounts 2"
    )

    return GenData(
        protocol=protocol.info,
        redteam=redteam.info,
        labels=unsupervised.labels,
        annotations=annotations,
        cue_amounts_1=cue_amounts_1,
        cue_amounts_2=cue_amounts_2,
        eval_tuples=unsupervised.eval_tuples,
        output_1_is_ref=unsupervised.output_1_is_ref,
        rt_meta=rt_meta,
        bt_meta=bt_meta,
        an_meta=an_meta,
        version=VERSION,
    )


class GenFtData(TypedDict):
    ft_id: str
    protocol: Metadata
    redteam: Metadata
    gen_data: GenData
    eval_tuples: list[EvalTuple]
    labels: list[bool]
    output_1_is_ref: list[bool]
    exploits: list[str]
    exp_meta: list[Metadata]
    exp_labels: list[bool]
    cues_amounts_e: list[float]
    cues_amounts_1: list[float]
    cues_amounts_2: list[float]
    annotations: list[bool]
    an_meta: list[Metadata]
    exp_annotations: list[bool]
    exp_an_meta: list[Metadata]
    post_annotations: list[bool]
    post_an_meta: list[Metadata]
    post_exp_annotations: list[bool]
    post_exp_an_meta: list[Metadata]
    version: str


async def gen_and_ft(
    protocol: Protocol,
    redteam: Redteam,
    n_supervised: int = 200,
    n_unsupervised: int = 400,
    n_supervised_test: int = 200,
):
    gen_data = await gen_labels(protocol, redteam, n_supervised, n_unsupervised)

    test_ds = await ComparisonDs.from_alpaca("val", range(n_supervised, n_supervised + n_supervised_test))

    exp_and_meta = await gather_tqdm(
        [redteam.exploit(t.instruction) for t in test_ds.eval_tuples], "Generating exploits"
    )
    exploits, exp_meta = [e[0] for e in exp_and_meta], [e[1] for e in exp_and_meta]

    cues_amounts_e = await gather_tqdm(
        [redteam.cue_amount(t.instruction, e) for t, e in zip(test_ds.eval_tuples, exploits)], "Getting cue amounts e"
    )
    cues_amounts_1 = await gather_tqdm(
        [redteam.cue_amount(t.instruction, t.output_1) for t in test_ds.eval_tuples], "Getting cue amounts 1"
    )
    cues_amounts_2 = await gather_tqdm(
        [redteam.cue_amount(t.instruction, t.output_2) for t in test_ds.eval_tuples], "Getting cue amounts 2"
    )

    exp_test_ds = await test_ds.replace_comparison(exploits)

    async def get_annotations(ds, model):
        annotator = get_simple_annotator(model)
        an_and_meta = await gather_tqdm([annotator(t) for t in ds.eval_tuples], "Annotating test")
        return [a[0] for a in an_and_meta], [a[1] for a in an_and_meta]

    trusted = DirectModel.trusted()
    annotations, an_meta = await get_annotations(test_ds, trusted)
    exp_annotations, exp_an_meta = await get_annotations(exp_test_ds, trusted)

    relabel_ds = ComparisonDs(gen_data["eval_tuples"], gen_data["annotations"], gen_data["output_1_is_ref"])
    ft_id = relabel_ds.hash()[:10]
    new_model = await train(relabel_ds, f"cp_{ft_id}", val_ds=test_ds, base_model=trusted.llm.model_ids[0])

    post_annotations, post_an_meta = await get_annotations(test_ds, new_model)
    post_exp_annotations, post_exp_an_meta = await get_annotations(exp_test_ds, new_model)

    return GenFtData(
        ft_id=ft_id,
        protocol=protocol.info,
        redteam=redteam.info,
        gen_data=gen_data,
        eval_tuples=test_ds.eval_tuples,
        labels=test_ds.labels,
        output_1_is_ref=test_ds.output_1_is_ref,
        exploits=exploits,
        exp_meta=exp_meta,
        exp_labels=exp_test_ds.labels,
        cues_amounts_e=cues_amounts_e,
        cues_amounts_1=cues_amounts_1,
        cues_amounts_2=cues_amounts_2,
        annotations=annotations,
        an_meta=an_meta,
        exp_annotations=exp_annotations,
        exp_an_meta=exp_an_meta,
        post_annotations=post_annotations,
        post_an_meta=post_an_meta,
        post_exp_annotations=post_exp_annotations,
        post_exp_an_meta=post_exp_an_meta,
        version=VERSION,
    )
