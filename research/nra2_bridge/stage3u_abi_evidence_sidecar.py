"""Research-only S3U token-span ABI sidecar serializer.

Pure additive evidence helper: never loads weights, trains, scores or alters S3U.
Caller must establish S3T physical provenance and 616 J7 before invoking.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_abi_evidence(
    output_dir, *, qmask, aligned, token_mask, token_lengths,
    prompt_sha256, provenance, j7_verified, prereg_verified,
):
    """Persist validated query ABI. Fail closed; output directory must be empty.

    provenance requires immutable physical hashes for S3T ZIP, inner JSON,
    original S3U FP16, and source revisions. No labels or logits are read.
    """
    if j7_verified is not True or prereg_verified is not True:
        raise AssertionError("S3U provenance and 616 J7 must precede ABI evidence")
    required = ("s3t_zip_sha256", "s3t_inner_json_sha256",
                "s3t_commitment_sha256", "s3u_token_fp16_sha256",
                "tokenizer_revision", "s3o_mask_builder_revision")
    if any(not isinstance(provenance.get(k), str) or not provenance[k]
           for k in required):
        raise AssertionError("Missing pinned source provenance")
    if provenance["s3t_commitment_sha256"] != "034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3":
        raise AssertionError("S3T prereg commitment drift")
    n = len(aligned)
    if n != 64 or len(prompt_sha256) != n or len(set(prompt_sha256)) != n:
        raise AssertionError("Expected 64 unique precommitted prompts")
    if any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
           for h in prompt_sha256):
        raise AssertionError("Invalid prompt SHA256")
    token_mask = token_mask.detach().cpu().to(torch.bool)
    token_lengths = token_lengths.detach().cpu().to(torch.int64)
    query = qmask.query_mask.detach().cpu().to(torch.bool)
    anchor = qmask.anchor_mask.detach().cpu().to(torch.bool)
    order = qmask.node_order.detach().cpu().to(torch.int64)
    if token_mask.ndim != 2 or token_mask.shape[0] != n:
        raise AssertionError("Invalid CPU token mask")
    t = token_mask.shape[1]
    if (query.shape != (n,t) or anchor.shape != (n,2,t)
            or order.shape != (n,2) or token_lengths.shape != (n,)):
        raise AssertionError("S3O ABI shape mismatch")
    spans = torch.empty((n,2,2),dtype=torch.int64)
    for i,a in enumerate(aligned):
        length = int(token_lengths[i])
        if not 0 < length <= t or int(a["tokens"]) != length:
            raise AssertionError("Tokenizer/CPU token length drift")
        if not torch.all(token_mask[i,:length]) or torch.any(token_mask[i,length:]):
            raise AssertionError("Non-contiguous CPU padding")
        if torch.any(query[i] & ~token_mask[i]) or torch.any(anchor[i] & ~query[i]):
            raise AssertionError("Query or anchor outside nonpadding query")
        mentions = [m for m in a["mentions"]
                    if m.token_positions and set(m.token_positions) ==
                    set(torch.where(anchor[i,0] | anchor[i,1])[0].tolist())]
        # Select anchors by their exact token positions; do not infer from gold.
        matched = []
        for k in range(2):
            pos = sorted(set(torch.where(anchor[i,k])[0].tolist()))
            if not pos or pos != list(range(pos[0],pos[-1]+1)):
                raise AssertionError("Empty or noncontiguous mention span")
            candidates = [m for m in a["mentions"]
                          if sorted(set(m.token_positions)) == pos]
            if len(candidates) != 1:
                raise AssertionError("Ambiguous native mention alignment")
            matched.append(candidates[0])
            spans[i,k] = torch.tensor((pos[0],pos[-1]+1))
        if (torch.any(anchor[i,0] & anchor[i,1])
                or len({(m.char_start,m.char_end) for m in matched}) != 2
                or [m.node for m in matched] != order[i].tolist()
                or not matched[0].char_start < matched[1].char_start):
            raise AssertionError("Mention order, identity or overlap drift")
    tensors = {
        "query_mask":query.contiguous(), "anchor_mask":anchor.contiguous(),
        "node_order":order.contiguous(),
        "mention_token_start_end":spans.contiguous(),
        "token_lengths":token_lengths.contiguous(),
    }
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("ABI evidence directory must be empty")
    out.mkdir(parents=True,exist_ok=True)
    sidecar = out / "NRA2_S3U_QUERY_ABI_EVIDENCE.safetensors"
    save_file(tensors,str(sidecar))
    reloaded = load_file(str(sidecar),device="cpu")
    if set(reloaded) != set(tensors) or any(
        not torch.equal(reloaded[k],v) for k,v in tensors.items()
    ):
        raise AssertionError("Physical sidecar reload mismatch")
    manifest = {
        "stage":"NRA2_S3U_ABI_EVIDENCE_ONLY",
        "output_gate":0, "candidate_accepted":False,
        "train_count":0, "candidate_scored":False,
        "fixed_hard_threshold":0.5,
        "prompt_sha256_in_order":list(prompt_sha256),
        "provenance":dict(provenance),
        "sidecar_sha256":_sha(sidecar),
        "tensor_schema":{k:{"dtype":str(v.dtype),"shape":list(v.shape)}
                         for k,v in tensors.items()},
    }
    (out / "NRA2_S3U_QUERY_ABI_MANIFEST.json").write_text(
        json.dumps(manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return manifest
