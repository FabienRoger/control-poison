# %%
from cpoison.main import GenData, VERSION
import json
from pathlib import Path
import numpy as np

data_folder = Path(__file__).parent.parent / "data"
runs: list[GenData] = [json.loads(file.read_text()) for file in data_folder.glob("*.json")]
runs = [r for r in runs if r["version"] == VERSION]

r = runs[0]
# %%
from tabulate import tabulate
from matplotlib import pyplot as plt

table = [[s.replace("_", "\n") for s in [
    "protocol",
    "redteam",
    "reg_acc_before",
    "reg_acc_after",
    "exp_acc_before",
    "exp_acc_after",
    "ref_pref",
    "ref_exp_pref",
    "pref_first_before",
    "pref_first_after",
]]]
for r in runs:
    reg_accuracy_before = np.mean([a == b for a, b in zip(r["annotations"], r["labels"])])
    reg_accuracy_after = np.mean([a == b for a, b in zip(r["post_annotations"], r["labels"])])
    exp_accuracy_before = np.mean([a == b for a, b in zip(r["exp_annotations"], r["exp_labels"])])
    exp_accuracy_after = np.mean([a == b for a, b in zip(r["post_exp_annotations"], r["exp_labels"])])
    ref_preference = np.mean([a == b for a, b in zip(r["output_1_is_ref"], r["labels"])])
    ref_exp_preference = np.mean([a == b for a, b in zip(r["output_1_is_ref"], r["exp_labels"])])
    pref_first_before = np.mean([a == 1 for a in r["annotations"]])
    pref_first_after = np.mean([a == 1 for a in r["post_annotations"]])

    table.append([
        r["protocol"]["name"],
        r["redteam"]["name"],
        reg_accuracy_before,
        reg_accuracy_after,
        exp_accuracy_before,
        exp_accuracy_after,
        ref_preference,
        ref_exp_preference,
        pref_first_before,
        pref_first_after,
    ])

print(tabulate(table, headers="firstrow"))
# %%
for i in range(20):
    print(r["output_1_is_ref"][i], r["eval_tuples"][i])
# %%
np.mean(r["output_1_is_ref"])
# %%
plt.hist([d["prop"] for d in runs[-1]["gen_data"]["an_meta"]])
# %%
