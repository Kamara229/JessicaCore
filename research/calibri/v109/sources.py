"""Pinned TRAIN parquet inputs. Calibration and labeled validation use separate datasets."""
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from activation_metric_v73 import train_activations
from .feedback import principal_directions


TRAIN_FILES = {
    "wiki": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
             "wikitext-2-raw-v1", "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"),
    "news": ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4",
             "data", "fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee"),
    "boolq": ("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5",
              "data", "4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623"),
    "sciq": ("allenai/sciq", "2c94ad3e1aafab77146f384e23536f97a4849815",
             "data", "19644360954006d06e9ad3df07bddb34f8535c081b831d48f604603c713ac167"),
}


def train_slice(dataset, start, count, columns=None):
    repo, revision, folder, digest = TRAIN_FILES[dataset]
    path = Path(hf_hub_download(repo, f"{folder}/train-00000-of-00001.parquet",
                                repo_type="dataset", revision=revision))
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"unexpected {dataset} training file")
    rows = pq.read_table(path, columns=columns).slice(start, count).to_pylist()
    if len(rows) != count:
        raise ValueError(f"incomplete {dataset} slice")
    return rows


def calibration(teacher, tokenizer):
    ids = []
    for ds, start in (("wiki", 1600), ("news", 2400)):
        rows = train_slice(ds, start, 32, ["text"])
        text = "\n".join(row["text"] for row in rows if row["text"].strip())
        part = tokenizer.encode(text, add_special_tokens=False,
                                truncation=True, max_length=128)
        if len(part) != 128:
            raise ValueError("calibration window too short")
        ids.extend(part)
    ids = torch.tensor(ids, dtype=torch.long)
    # q/k/v share one input, gate/up another. Hook the actual trained operator once.
    representatives = {}
    for name, layer in teacher.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        key = name.replace(".k_proj", ".q_proj").replace(".v_proj", ".q_proj")
        key = key.replace(".up_proj", ".gate_proj")
        weight = "model.embed_tokens.weight" if name == "lm_head" else name + ".weight"
        representatives[weight] = key
    modules = {name: teacher.get_submodule(name) for name in sorted(set(representatives.values()))}
    activations = train_activations(teacher, modules, ids)
    bases = {key: principal_directions(x) for key, x in activations.items()}
    return ({name: bases[key] for name, key in representatives.items()},
            hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest())
