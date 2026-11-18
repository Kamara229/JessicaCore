"""S68: actual original S95 source-level future-conditioning forensic, research-only.

Read the exact original S95 prompt_only_vector function via Python AST and run it
without importing training/model modules. S95 hashes *all* source question tokens,
token positions and all character trigrams. Full-prompt hashed features therefore
can encode future suffix tokens when used to condition an earlier hidden state.

This study uses ONLY already-spent S66 raw Russian strings and a deterministic
byte-token *surrogate* (NOT original Calibri BPE). It proves a source-level
non-prefix-invariance of the original S95 feature algorithm, NOT a physical
12-layer GT leak measurement or production autoregressive fault.
No model, adapter weights, labels, gradients, routes or thresholds loaded.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

S95_PATH=Path("research/calibri2/s95/train_and_audit_router.py")
EXPECT_DIM=512
COUNTERFACTUAL_A=" После этого никто не передавал документы."
COUNTERFACTUAL_B=" Затем документы были переданы получателю."


def require(ok,why):
    if not bool(ok):
        raise ValueError("S68 fail-closed S95 future feature audit: "+why)


class ByteTokenSurrogate:
    """Explicitly NOT original BPE; only used to execute frozen S95 AST."""
    def encode(self,text,add_special_tokens=False):
        require(isinstance(text,str) and not add_special_tokens,
                "source-only surrogate")
        return list(text.encode("utf-8"))


def original_s95_function_from_ast(path=S95_PATH):
    """Run unmodified S95 function body, not a reimplementation."""
    source=Path(path).read_text(encoding="utf-8")
    tree=ast.parse(source,filename=str(path))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)
           and n.name=="prompt_only_vector"]
    require(len(nodes)==1, "exactly one original S95 prompt-only function")
    fn=nodes[0]
    arguments=[a.arg for a in fn.args.args]
    require(arguments==["tok","question"] and not fn.decorator_list,
            "S95 signature/decorator drift")
    dimensions=[n for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=="DIM"
                        for t in n.targets)]
    require(len(dimensions)==1 and isinstance(dimensions[0].value,ast.Constant)
            and dimensions[0].value.value==EXPECT_DIM,
            "S95 hashed dimension or implementation drift")
    # No extra imports or unrelated module initialization are evaluated.
    only_fn=ast.Module(body=[fn],type_ignores=[])
    ast.fix_missing_locations(only_fn)
    scope={"np":np,"zlib":zlib,"DIM":EXPECT_DIM}
    exec(compile(only_fn,str(path),"exec"),scope)
    f=scope["prompt_only_vector"]
    return f,hashlib.sha256(source.encode("utf-8")).hexdigest(),hashlib.sha256(
        ast.dump(fn,include_attributes=False).encode("utf-8")).hexdigest()


def vector_pair_drift(fn,tok,source,cut):
    require(isinstance(source,str) and isinstance(cut,int)
            and 4<=cut<=len(source)-4, "source-only boundary range")
    prefix=source[:cut]
    full=fn(tok,source)
    truncated=fn(tok,prefix)
    counter_a=fn(tok,prefix+COUNTERFACTUAL_A)
    counter_b=fn(tok,prefix+COUNTERFACTUAL_B)
    require(all(isinstance(z,np.ndarray) and
                z.shape==(EXPECT_DIM,) and np.all(np.isfinite(z))
                for z in (full,truncated,counter_a,counter_b)),
            "exact S95 emitted nonfinite/wrong shape")
    def measure(a,b):
        d=np.abs(a.astype(np.float64)-b.astype(np.float64))
        return {"max_abs":float(np.max(d)),
                "mean_abs":float(np.mean(d)),
                "exact_equal":bool(np.array_equal(a,b)),
                "cosine":float(np.dot(a,b) / max(
                    1.e-12, float(np.linalg.norm(a)*np.linalg.norm(b))))}
    return {
        "native_calibri_tokenizer_used":False,
        "surrogate_only_no_native_H8":True,
        "source_length_chars":len(source),
        "source_sha256":hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "prefix_length_chars":len(prefix),
        "prefix_sha256":hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "full_vs_prefix_prompt_features":measure(full,truncated),
        "shared_prefix_suffixA_vs_suffixB_features":
            measure(counter_a,counter_b),
        "source_only_no_factual_label_or_inference":True,
    }


def source_word_mid_cut(text):
    """Surface-only ASCII-space word cut; NOT asserted native BPE safe."""
    require(text and " " in text and len(text)>16,"Russian raw prompt")
    positions=[i for i,c in enumerate(text) if c==" "
               and 4<=i<=len(text)-4]
    require(positions,"no source-only mid-word boundary")
    return min(positions,key=lambda i:abs(i-len(text)//2))


def audit_spent_raw_only(raw_inputs):
    require(len(raw_inputs)==32 and len(set(raw_inputs))==32,
            "exact already-spent S66 32 inputs (NO gold)")
    f,source_sha,ast_sha=original_s95_function_from_ast()
    tok=ByteTokenSurrogate()
    rows=[vector_pair_drift(f,tok,text,source_word_mid_cut(text))
          for text in raw_inputs]
    return {
        "stage":"NRA3_S68_S95_ORIGINAL_SOURCE_ALGORITHM_PREFIX_FUTURE_ABLATION",
        "S95_file_sha256":source_sha,"S95_function_AST_sha256":ast_sha,
        "original_S95_python_function_unmodified_AST_executed":True,
        "native_calibri_BPE_or_real_43M_H8_test":False,
        "byte_token_surrogate_only_NOT_native":True,
        "already_spent_S66_RAW_only":True,
        "n_rows":len(rows),
        "n_full_vs_prefix_feature_change":sum(
            not x["full_vs_prefix_prompt_features"]["exact_equal"] for x in rows),
        "n_equal_prefix_suffix_swap_feature_change":sum(
            not x["shared_prefix_suffixA_vs_suffixB_features"]["exact_equal"]
            for x in rows),
        "future_features_structurally_present_in_original_S95":True,
        "causal_original_streaming_root_cause_or_safety_proven":False,
        "old_original_calibri_or_GT_modified":False,
        "no_fresh_held_or_EVAL32_128_or_gold":True,
        "no_optimizer_no_threshold_selection_no_adapter_predictions":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
        "rows":rows,
    }


def main():
    import argparse
    from research.nra3_s20.s66_frozen_new32_raw_only_DO_NOT_TRAIN import (
        raw_only_inputs,raw_sha256,
    )
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    data=raw_only_inputs()
    r=audit_spent_raw_only(data)
    r["precommitted_S66_raw_text_SHA256"]=raw_sha256()
    require(not args.out.exists(),"research report immutable")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("S68_S95_FROZEN_SOURCE_FUTURE_FEATURE_COUNTERFACTUAL",
          json.dumps({k:r[k] for k in (
            "n_rows","n_full_vs_prefix_feature_change",
            "n_equal_prefix_suffix_swap_feature_change",
            "native_calibri_BPE_or_real_43M_H8_test",
            "output_gate")},sort_keys=True))


if __name__=="__main__":
    main()
