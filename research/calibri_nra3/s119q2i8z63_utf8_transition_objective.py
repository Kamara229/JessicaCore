"""GPT-only UTF8 transition loss, source/tokenizer preflight.
Train on legal reference prefixes only. Penalize probability mass on
byte-invalid continuations; NEVER mask inference logits. No new weights.
"""
import codecs
import hashlib
import json
from collections import Counter
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import BYTE_PIECE_PATTERN,native_bytelevel_full_utf8_status
from research.calibri_nra3.s119q2i8z60_native_pair_objective import native_plan
from research.calibri_nra3.s119q2i8z59_gpt_compositional_curriculum import curriculum
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import verified_native_answer
PLAN_SHA="2db9db9b5e4609e2619fa48e4046d4cd65f9257d2c8ee6d0eb093fb042d9f3cc"
CURRICULUM_SHA="fb68dde68942c9b4b686e88e64a7916d6152fd11e853de989495315fc6de93a5"
WEIGHT=.25
def require(ok,msg):
    if not ok:raise ValueError(msg)
def reverse_byte_map():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256))
    cs=list(bs);n=0
    for b in range(256):
        if b not in bs:
            bs.append(b);cs.append(256+n);n+=1
    return {chr(c):b for b,c in zip(bs,cs)}
def byte_pieces(tok):
    reverse=reverse_byte_map();special=set(tok.all_special_ids)
    result=[]
    for i in range(tok.vocab_size):
        p=str(tok.convert_ids_to_tokens(i))
        if i in special or (p.startswith("<|") and p.endswith("|>")):
            result.append(None);continue
        m=BYTE_PIECE_PATTERN.fullmatch(p)
        if m:
            result.append(bytes([int(m.group(1),16)]));continue
        result.append(bytes(reverse[x] for x in p) if p and all(x in reverse for x in p) else None)
    return result
def advance(pending,piece):
    if piece is None:return None
    dec=codecs.getincrementaldecoder("utf-8")("strict")
    dec.setstate((pending,0))
    try:
        dec.decode(piece,final=False)
        return dec.getstate()[0]
    except UnicodeDecodeError:
        return None
def allowed_mask(pieces,pending,eos):
    # EOS only valid at complete bytes, other special tokens forbidden.
    return torch.tensor([not pending if i==eos else advance(pending,p) is not None
                         for i,p in enumerate(pieces)],dtype=torch.bool)
def invalid_mass_loss(logits,mask,*,normalizer=1,weight=WEIGHT):
    require(logits.ndim==2 and logits.shape[0]==1 and mask.ndim==1 and
            logits.shape[1]==mask.numel() and mask.dtype==torch.bool and
            bool(mask.any()) and bool((~mask).any()) and normalizer>0 and
            bool(torch.isfinite(logits).all()),"finite logits and nonempty valid/invalid sets")
    mask=mask.to(logits.device)
    x=logits[0].float()
    # -log P(valid); differentiable through all logits, not output filtering.
    return weight*F.softplus(torch.logsumexp(x[~mask],dim=0)-torch.logsumexp(x[mask],dim=0))/normalizer
def plan(tok):
    manifest_check=native_plan(tok)
    require(manifest_check["native_plan_SHA256"]==PLAN_SHA and manifest_check["curriculum_SHA256"]==CURRICULUM_SHA,
            "exact pinned Z60 token plan")
    rows=[verified_native_answer(tok,r) for r in curriculum()]
    require(tok.vocab_size==49152 and tok.eos_token_id==2,"original native tokenizer")
    pieces=byte_pieces(tok);cache={};counts=Counter();positions=0;pending_count=0
    manifest=[]
    for row in rows:
        gold=row["answer_native_with_eos"];pending=b"";prior=[];states=[]
        for t,target in enumerate(gold):
            native=native_bytelevel_full_utf8_status(tok,prior)["entire_original_native_BPE_prefix_UTF8_status"]
            require(native==("INCOMPLETE_UTF8_NEEDS_CONTINUATION" if pending else "VALID_COMPLETE_UTF8"),
                    "independent reference byte reconstruction parity")
            if pending not in cache:
                cache[pending]=allowed_mask(pieces,pending,tok.eos_token_id)
            mask=cache[pending]
            require(bool(mask[target]) and bool(mask.any()) and bool((~mask).any()),
                    "every authored target admitted; no empty sets")
            key=pending.hex();states.append(key);counts[key]+=1
            positions+=1;pending_count+=int(bool(pending))
            if target==tok.eos_token_id:
                require(not pending and t==len(gold)-1,"terminal EOS at complete UTF8")
            else:
                pending=advance(pending,pieces[target])
                require(pending is not None,"all authored target byte transitions valid")
            prior.append(target)
        manifest.append({"id":row["id"],"states":states})
    require(positions==1618 and len(rows)==98,"exact frozen Z60 TRAIN native plan")
    mask_stats={p.hex():{"positions":counts[p.hex()],"allowed":int(m.sum()),
                "mask_SHA256":hashlib.sha256(bytes(m.tolist())).hexdigest()}
                for p,m in sorted(cache.items())}
    return {"stage":"Z63_GPT_UTF8_TRANSITION_OBJECTIVE_PREFLIGHT",
        "native_plan_SHA256":PLAN_SHA,"curriculum_SHA256":CURRICULUM_SHA,
        "state_plan_SHA256":hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "train_rows":98,"target_positions":positions,"pending_positions":pending_count,
        "unique_UTF8_states":len(cache),"state_masks":mask_stats,"weight":WEIGHT,
        "all_reference_targets_admitted":True,"inference_masking":False,
        "model_forwards":0,"optimizer_steps":0,"FP32_calls":0,"new_parameters":0,
        "candidate_accepted":False,"output_gate":0}
if __name__=="__main__":
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    print("Z63_UTF8_PLAN",json.dumps(plan(tok),sort_keys=True),flush=True)
