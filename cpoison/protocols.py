from abc import ABC, abstractmethod
from lowstakes.utils import AndMetadata, WithInfo
from cpoison.eval import EvalTuple
from cpoison.data import ComparisonDs
from cpoison.annotators import Annotator
from cpoison.models import Model, DirectModel
from attrs import define
from cpoison.annotators import get_simple_annotator


class Protocol(ABC, WithInfo):
    @abstractmethod
    def get_annotator(
        self, supervised: ComparisonDs, unsupervised: list[EvalTuple], untrusted: Model
    ) -> AndMetadata[Annotator]:
        ...


class UseTrusted(Protocol):
    def get_annotator(
        self, supervised: ComparisonDs, unsupervised: list[EvalTuple], untrusted: Model
    ) -> AndMetadata[Annotator]:
        return get_simple_annotator(DirectModel.trusted()), {}


class UseUntrusted(Protocol):
    def get_annotator(
        self, supervised: ComparisonDs, unsupervised: list[EvalTuple], untrusted: Model
    ) -> AndMetadata[Annotator]:
        return get_simple_annotator(untrusted), {}
