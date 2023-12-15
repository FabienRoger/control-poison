# %%
from cpoison.main import GenData
import json
from pathlib import Path
import numpy as np

data_folder = Path(__file__).parent.parent / "data"
runs: list[GenData] = [json.loads(file.read_text()) for file in data_folder.glob("*.json")]

r = runs[0]
# %%
for r in runs:
    print(r["protocol"]["name"], r["redteam"]["name"])
    reg_accuracy_before = np.mean([a == b for a, b in zip(r["annotations"], r["labels"])])
    print(f"{reg_accuracy_before=}")
    reg_accuracy_after = np.mean([a == b for a, b in zip(r["post_annotations"], r["labels"])])
    print(f"{reg_accuracy_after=}")
    exp_accuracy_before = np.mean([a == b for a, b in zip(r["exp_annotations"], r["exp_labels"])])
    print(f"{exp_accuracy_before=}")
    exp_accuracy_after = np.mean([a == b for a, b in zip(r["post_exp_annotations"], r["exp_labels"])])
    print(f"{exp_accuracy_after=}")
    ref_preference = np.mean([a == b for a, b in zip(r["output_1_is_ref"], r["labels"])])
    print(f"{ref_preference=}")
    ref_exp_preference = np.mean([a == b for a, b in zip(r["output_1_is_ref"], r["exp_labels"])])
    print(f"{ref_exp_preference=}")
# %%
for i in range(20):
    print(r["output_1_is_ref"][i], r["eval_tuples"][i])
# %%
np.mean(r["output_1_is_ref"])
# %%
