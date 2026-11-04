"""Constrained labeled QA on pinned, paired train holdouts (separate from free chat)."""
import hashlib
import random

import torch
from v94.answers import summary
from v94.qa import encode_case
from v108.generation import build_chat
from .protocol import QA_SPANS
from .sources import train_slice


def cases(tok, domain):
    start, count = QA_SPANS[domain]
    rows = train_slice(domain, start, count)
    labels = ("Yes", "No") if domain == "boolq" else ("A", "B", "C", "D")
    variants = []
    for label in labels:
        alternatives = [tok.encode(value, add_special_tokens=False)
                        for value in (label, " " + label)]
        single = sorted({ids[0] for ids in alternatives if len(ids) == 1})
        if not single or any(set(single).intersection(group) for group in variants):
            raise ValueError("answer labels have missing or overlapping first-token IDs")
        variants.append(single)

    records = []
    for index, row in enumerate(rows, start):
        if domain == "boolq":
            context = row["passage"]
            instruction = "Reply Yes or No."
            gold = 0 if row["answer"] else 1
        else:
            context = row["support"]
            correct = row["correct_answer"]
            options = [correct] + [row[f"distractor{i}"] for i in range(1, 4)]
            if len(set(options)) != 4:
                raise ValueError(f"ambiguous SciQ options for training row {index}")
            seed = hashlib.sha256(f"v109/{index}".encode()).digest()
            random.Random(seed).shuffle(options)
            instruction = "\n".join(f"{k}) {v}" for k, v in zip(labels, options))
            instruction += "\nReply with the letter of the correct option."
            gold = options.index(correct)
        # Retain the whole short passage, question and every answer option.
        prompt = f"{context}\nQuestion: {row['question']}\n{instruction}"
        records.append(encode_case(tok, build_chat(tok, prompt), index, gold))
    return records, variants


@torch.inference_mode()
def score(model, records, variants):
    rows = []
    for case in records:
        tokens = torch.tensor([case["tokens"]], dtype=torch.long)
        logits = model(input_ids=tokens, use_cache=False).logits[0, -1]
        # Marginalize the two possible leading-space tokenizations of each label.
        masses = torch.stack([logits[ids].logsumexp(0) for ids in variants])
        pred = int(masses.argmax())
        rows.append({k: case[k] for k in ("record_index", "prompt_SHA", "gold")}
                    | {"pred": pred, "correct": int(pred == case["gold"])})
    return {"summary": summary(rows, len(variants)), "rows": rows}
