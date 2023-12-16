from cpoison.data import ComparisonDs
from cpoison.base_models import DirectModel
from cpoison.annotators import get_simple_annotator
from lowstakes.utils import gather_tqdm


async def trusted_accuracy(supervised: ComparisonDs) -> float:
    annotator = get_simple_annotator(DirectModel.trusted())
    an_and_meta = await gather_tqdm([annotator(t) for t in supervised.eval_tuples], "Annotating supervised")
    annotations = [a[0] for a in an_and_meta]
    corrects = sum(a == l for a, l in zip(annotations, supervised.labels))
    return sum(corrects) / len(supervised.labels)
