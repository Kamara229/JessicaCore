"""Frozen SmolLM2 INT6/CB59 four-way and layer/factor boundary diagnosis."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from shared_checkpoint_v58 import pack_shared, cold_load_shared
from v84.sources import original_teacher
from v94.generation import generate_one
from v108.generation import build_chat
from .mix import swap, permutations, distributions, compare
from .protocol import PROBES, CUTS, FACTORS, CB59_BYTES, INT6_BYTES, validate_geometry
from v86.provenance import ORIGINAL_CB59_SHA

def load_controls(archive, out):
    source = archive.read_bytes()
    if len(source) != CB59_BYTES or hashlib.sha256(source).hexdigest() != ORIGINAL_CB59_SHA:
        raise AssertionError("CB59 original immutable physical source mismatch")
    packed, meta, _ = make_model("baseline", SimpleNamespace(original=archive, folder=out))
    int6 = original_teacher()
    # Independent physical INT6 codec, NOT faked FP32 teacher or unquantized layer.
    with TemporaryDirectory(dir=out) as folder:
        p = Path(folder) / "control.cb58"
        blob, *_ = pack_shared(int6.state_dict())
        if len(blob) != INT6_BYTES:
            raise AssertionError(f"expected physically paid INT6 archive, got {len(blob)}")
        sha = hashlib.sha256(blob).hexdigest()
        p.write_bytes(blob)
        physical = cold_load_shared(int6, p)
        del blob
    if physical["actual_entire_checkpoint_bytes"] != INT6_BYTES:
        raise AssertionError("INT6 actually restored incomplete source")
    if int6.model.embed_tokens.weight.data_ptr() != int6.lm_head.weight.data_ptr():
        raise AssertionError("INT6 source is not tied")
    gc.collect()
    return packed, int6, {"CB59":meta, "INT6":physical, "INT6_SHA256":sha}

def probes(tok):
    return [{"domain": domain, "question": question, "gold": gold,
             "chat": build_chat(tok, question),
             "question_sha256": hashlib.sha256(question.encode()).hexdigest()}
            for domain, question, gold in PROBES]

def summarize_text(tok, output):
    tokens = output["ids"]
    cut = tokens.index(tok.eos_token_id) if tok.eos_token_id in tokens else len(tokens)
    content = tok.decode(tokens[:cut], skip_special_tokens=True)
    return {"first_turn_text": content,
            "first_turn_ids": tokens[:cut],
            "first_turn_eos": cut < len(tokens),
            "first_turn_token_count": cut,
            "timing_seconds": output["elapsed_seconds"],
            "no_semantic_pass_claim": True}

def run(archive, out):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    packed, int6, physical = load_controls(archive, out)
    validate_geometry(len(packed.model.layers))
    prompts = probes(tok)
    contexts = [torch.tensor([tok.encode(p["chat"], add_special_tokens=False)],
                             dtype=torch.long) for p in prompts]
    # Use the physical INT6 control as a good-answer *reference*, not an
    # assumed truth oracle. Independent factual gold is only for later review.
    reference = [distributions(int6, x) for x in contexts]
    records = {"revision": MODEL_SHA, "control_physical": physical,
               "diagnostic_hybrids_unpaid_not_candidates": True,
               "original_v89_TEST_not_opened": True,
               "prompts": [{k:v for k,v in p.items() if k!="chat"} for p in prompts],
               "four_way": {}, "cumulative_prefix": {}, "factor_whole_network": {}}
    for name, head, blocks in permutations():
        with swap(packed, int6, head=head, block_ids=blocks):
            rows = []
            for i, (p,x) in enumerate(zip(prompts, contexts)):
                logits = distributions(packed, x)
                answer = generate_one(packed, tok, p["chat"], stop_at_eos=True)
                rows.append({"prompt_index":i, **compare(reference[i],logits),
                             **summarize_text(tok, answer)})
            records["four_way"][name] = rows
        print(json.dumps({"experiment":"v110_four_way", "mode":name,
                          "overlap":[r["top10_overlap"] for r in rows],
                          "first_turns":[r["first_turn_text"][:90] for r in rows]}),flush=True)

    # Diagnostic cumulative sweeps do not select or fit any candidate.
    for cutoff in CUTS:
        with swap(packed, int6, block_ids=range(cutoff)):
            rows = [compare(reference[i], distributions(packed, x))
                    for i, x in enumerate(contexts[:3])]
        records["cumulative_prefix"][str(cutoff)] = rows
        print(json.dumps({"experiment":"v110_prefix_cut", "cut":cutoff,
                          "top10":[r["top10_overlap"] for r in rows]}),flush=True)

    # Attention-only and MLP-only full swaps; both directions keep head CB59.
    for factor in FACTORS:
        with swap(packed,int6,block_ids=range(30),factor=factor):
            rows=[compare(reference[i],distributions(packed,x))
                  for i,x in enumerate(contexts[:3])]
        records["factor_whole_network"][factor]=rows
        print(json.dumps({"experiment":"v110_factor", "factor":factor,
                          "top10":[r["top10_overlap"] for r in rows]}),flush=True)

    # Prove after all swaps no INT6 trained dense 2D module remains grafted.
    from direct_runtime_v59 import PackedLinear, PackedEmbedding
    if not isinstance(packed.model.embed_tokens, PackedEmbedding) or not isinstance(packed.lm_head,PackedLinear):
        raise AssertionError("original native tied head not restored")
    if any(isinstance(layer.self_attn.q_proj, torch.nn.Linear) for layer in packed.model.layers):
        raise AssertionError("original native transformer not restored")
    report=out/"v110_localization.json"
    report.write_text(json.dumps(records, ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"experiment":"v110_done","report":str(report),
                      "hybrid_not_a_paid_compact_model":True,
                      "no_NEW_weights":True}),flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.archive,a.out)
