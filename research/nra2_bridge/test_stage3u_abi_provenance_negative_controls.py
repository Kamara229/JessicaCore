"""S3U ABI negative controls: synthetic only, never read weights, labels or gold.

These tests document presently missing physical/text provenance validation.
They are strict xfails until the sidecar API accepts trusted physical sources.
"""
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import pytest
import torch
from research.nra2_bridge.stage3u_abi_evidence_sidecar import write_abi_evidence

COMMIT = "034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3"

@dataclass
class Mention:
    token_positions: list
    node: int
    char_start: int
    char_end: int

def synthetic():
    n, t = 64, 8
    query = torch.zeros(n,t,dtype=torch.bool)
    query[:,3:7] = True
    anchor = torch.zeros(n,2,t,dtype=torch.bool)
    anchor[:,0,3:5] = True
    anchor[:,1,5:7] = True
    return dict(
        qmask=SimpleNamespace(query_mask=query,anchor_mask=anchor,node_order=torch.tensor([[1,2]]*n)),
        aligned=[{"tokens":t,"mentions":[Mention([0],9,0,1),Mention([3,4],1,30,34),Mention([5,6],2,35,39)]} for _ in range(n)],
        token_mask=torch.ones(n,t,dtype=torch.bool),
        token_lengths=torch.full((n,),t,dtype=torch.long),
        prompt_sha256=[f"{i:064x}" for i in range(n)],
        provenance={"s3t_zip_sha256":"a"*64,"s3t_inner_json_sha256":"b"*64,
                    "s3t_commitment_sha256":COMMIT,"s3u_token_fp16_sha256":"c"*64,
                    "tokenizer_revision":"synthetic","s3o_mask_builder_revision":"synthetic"},
        j7_verified=True,prereg_verified=True,
    )

@pytest.mark.xfail(strict=True,reason="ordered prompt hashes are not derived from committed S3T prompt bytes")
def test_fabricated_but_well_formed_prompt_hashes_must_fail_closed():
    args=synthetic()
    args["prompt_sha256"]=[f"{i+100:064x}" for i in range(64)]
    with TemporaryDirectory() as out:
        with pytest.raises(AssertionError):
            write_abi_evidence(out,**args)

@pytest.mark.xfail(strict=True,reason="native char offsets are not cross-checked with committed text/tokenizer offsets")
def test_shifted_nonoverlapping_character_offsets_must_fail_closed():
    args=synthetic()
    for a in args["aligned"]:
        a["mentions"][1].char_start += 100
        a["mentions"][1].char_end += 100
        a["mentions"][2].char_start += 100
        a["mentions"][2].char_end += 100
    with TemporaryDirectory() as out:
        with pytest.raises(AssertionError):
            write_abi_evidence(out,**args)

@pytest.mark.xfail(strict=True,reason="external booleans and format-only hashes do not authenticate physical artifacts")
def test_fabricated_physical_sha_with_true_flags_must_fail_closed():
    args=synthetic()
    with TemporaryDirectory() as out:
        with pytest.raises(AssertionError):
            write_abi_evidence(out,**args)
