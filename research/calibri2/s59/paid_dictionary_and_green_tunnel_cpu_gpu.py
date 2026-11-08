"""CALIBRI S59: CPU/GPU-identical-method packed-dictionary + Green Tunnel rehearsal.

This is a TRAIN-only candidate generator, not a claim of improved language
quality or FP32 parity. No inspected/sealed QA (S26/S29/S53/S55/S57) is
imported or used to tune anything. Physical output is not promoted without
independent full-answer, old-skill and <=0.7 bpp checks.

To continue after Kaggle GPU quota ends, use --device cpu --resume path.
No GPU-only tensors/operators are needed by the packed inference contract.
"""
from __future__ import annotations
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import HfApi
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from v176.operators import clear
from v185.codec import replace_discrete
from v186.pairs import build_pairs

# Sealed, physically paid starting point: S56.
BASE_SOURCE_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
BASE_DUAL_SHA="503926839336ed7161786fa6d8ccb2a4ae6fee876cb566d0e52db80ff128f6d4"
BASE_MANIFEST_SHA="bb15a2638f9beb33318bbd97e156973043b220e8d8cda81874a69b61798d7c2b"
BASE_TOTAL=3181257
DATASET="Salesforce/wikitext"
CONFIG="wikitext-2-raw-v1"
SEED=59059

def sha(data): return hashlib.sha256(data).hexdigest()

def safe_device(name):
    if name=="auto": return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but physical GPU not present; use --device cpu")
    return torch.device("cuda:0" if name=="cuda" else "cpu")

def paid_manifest(src,side,manifest):
    n=SOURCE_PARAMS
    base={"stage":"S59_PACKED_DICTIONARY_AND_GREEN_TUNNEL_CANDIDATE",
          "source_sha256":sha(src),"controller_sha256":sha(side),
          "source_bytes":len(src),"controller_bytes":len(side),
          "original_architecture_params":n,"no_extra_weights":True,
          "CPU_portable":True,"not_promoted_without_real_quality_guard":True}
    for _ in range(20):
        text=json.dumps(base,sort_keys=True,indent=2).encode()
        total=len(src)+len(side)+len(text)
        nextv={"manifest_bytes":len(text),"all_inclusive_bytes":total,
               "all_inclusive_bpp":f"{(8*total/n):.12f}"}
        if all(base.get(k)==v for k,v in nextv.items()):break
        base.update(nextv)
    content=json.dumps(base,sort_keys=True,indent=2).encode()
    if len(src)+len(side)+len(content)!=base["all_inclusive_bytes"]:
        raise RuntimeError("all-inclusive paid manifest did not converge")
    if (8*base["all_inclusive_bytes"]/n)>.7:
        raise RuntimeError("0.7 bpp physical limit exceeded")
    manifest.write_bytes(content)
    return base

def frozen_train_source(tok,out,max_windows,context):
    """Only public TRAIN text. Hash/source kept for CPU checkpoint continuity."""
    idfile=out/"wiki_train_tokens.pt"
    metapath=out/"wiki_train_provenance.json"
    if idfile.is_file():
        assert metapath.is_file()
        ids=torch.load(idfile,map_location="cpu",weights_only=True)
        provenance=json.loads(metapath.read_text())
    else:
        revision=HfApi().dataset_info(DATASET).sha
        data=load_dataset(DATASET,CONFIG,split="train",revision=revision,
                          trust_remote_code=False)
        raw="\n\n".join(r["text"] for r in data if r["text"].strip())
        ids=torch.tensor(tok.encode(raw,add_special_tokens=False),dtype=torch.long)
        provenance={"dataset":DATASET,"subset":CONFIG,"split":"train",
                    "revision_sha":revision,"tokens":len(ids),
                    "dataset_role":"TRAIN_ONLY_NO_QA"}
        torch.save(ids,idfile)
        metapath.write_text(json.dumps(provenance,indent=2))
    if len(ids)<context*(max_windows+1):
        raise ValueError("Not enough TRAIN text for fixed deterministic windows")
    provenance["token_sha256"]=sha(ids.numpy().tobytes())
    metapath.write_text(json.dumps(provenance,indent=2))
    return ids,provenance

def example_loss(model,sources,first,later,row,device):
    """Full target sequence replay, including binary Yes/No and multi-token old skills."""
    targets=row["targets"].tolist()
    tokens=row["ids"][:-1].to(device)
    positions=row["positions"].tolist()
    if len(positions)!=len(targets) or not positions:
        raise ValueError("Malformed TRAIN next-token positions")
    clear(sources)
    out=model(input_ids=tokens[None],use_cache=False,output_hidden_states=True)
    hidden=out.hidden_states[-1][0]
    raw=out.logits[0]
    paid_head=model.lm_head.source()
    losses=[]
    for k,(pos,target) in enumerate(zip(positions,targets)):
        adapter=first if k==0 else later
        logits=raw[pos]+F.linear(adapter(hidden[pos]),paid_head)
        gold=logits[int(target)]
        competitor=logits.detach().clone()
        competitor[int(target)]=-1e9
        # Actual global-winner supervised margin on verified TRAIN label.
        loss=F.softplus(competitor.max()-gold+.2)
        losses.append(loss if k==0 else .75*loss)
    return torch.stack(losses).mean()

def text_loss(model,sources,later,ids,device):
    clear(sources)
    out=model(input_ids=ids[None],use_cache=False,output_hidden_states=True)
    paid_head=model.lm_head.source()
    # Model-free-form language TRAIN: later gate participates in *every*
    # causal hidden position rather than memorizing answer tables.
    delta=F.linear(later(out.hidden_states[-1][0,:-1]),paid_head)
    logits=out.logits[0,:-1]+delta
    return F.cross_entropy(logits,ids[1:])

def probe_native_cpu(archive_blob,dual_bytes,tok,original,augmented,old):
    """Native actual token sequences, no teacher-forced quality credit."""
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        f=Path(d)/"controller.safetensors";f.write_bytes(dual_bytes)
        state=load_file(str(f))
    model,sources=create_small_paid_model(CB59(archive_blob),build_config(MODEL_SPEC))
    model.requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,12)
    first.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("later.")})
    first.eval();later.eval()
    return (s56.probe(model,sources,first,later,head,tok,original),
            s56.probe(model,sources,first,later,head,tok,augmented),
            s56.probe(model,sources,first,later,head,tok,old[:4]))

def quality(rows,pairs):
    wins={k for k,v in rows.items() if v["full_correct"]}
    firsts={k for k,v in rows.items() if v["first_gold_correct"]}
    both=sum((f'{p["pair_id"]}_left' in wins and
              f'{p["pair_id"]}_right' in wins) for p in pairs)
    return wins,firsts,both

def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("controller",type=Path)
    p.add_argument("parent_manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--steps",type=int,default=24)
    p.add_argument("--context",type=int,default=48)
    p.add_argument("--windows",type=int,default=128)
    p.add_argument("--checkpoint-every",type=int,default=8)
    p.add_argument("--centroid-lr",type=float,default=.00004)
    p.add_argument("--tunnel-lr",type=float,default=.0005)
    p.add_argument("--lm-weight",type=float,default=.5)
    p.add_argument("--old-replay-weight",type=float,default=2.0)
    p.add_argument("--resume",type=Path,default=None)
    a=p.parse_args()
    if not 1<=a.steps<=1000:raise ValueError("Bound training to <=1000 steps")
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(SEED)
    device=safe_device(a.device)
    source=a.source.read_bytes()
    dual=a.controller.read_bytes()
    manifest=a.parent_manifest.read_bytes()
    if (len(source)!=SOURCE_BYTES or sha(source)!=BASE_SOURCE_SHA or
        sha(dual)!=BASE_DUAL_SHA or sha(manifest)!=BASE_MANIFEST_SHA or
        json.loads(manifest)["all_inclusive_total_bytes"]!=BASE_TOTAL):
        raise ValueError("Refuse to train on altered sealed S45/S56 physical artifacts")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    original=s30.make_training_pairs(tok)
    augmented=s54.build_new_train(tok)
    old=build_pairs(tok,"TRAIN")
    assert len(original)==32 and len(augmented)==32 and len(old)==24
    ids,provenance=frozen_train_source(tok,a.out,a.windows,a.context)
    archive=CB59(source)
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    model.to(device)
    model.eval()
    for q in sources.values():
        q.centroids.requires_grad_(True)
        q.log_scales.requires_grad_(False)
    with Path(a.controller).open("rb") as f:
        pass
    state=load_file(str(a.controller))
    first=s47.ActiveGreenTunnel(320,2).to(device)
    later=s47.ActiveGreenTunnel(320,12).to(device)
    first.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("later.")})
    first.eval().requires_grad_(False)
    later.train()
    paid_centroids=[q.centroids for q in sources.values()]
    optimizer=torch.optim.AdamW([
        {"params":paid_centroids,"lr":a.centroid_lr},
        {"params":list(later.parameters()),"lr":a.tunnel_lr}
    ],weight_decay=.00001)
    start=0
    if a.resume:
        saved=torch.load(a.resume,map_location="cpu",weights_only=True)
        if (saved["source_sha"]!=BASE_SOURCE_SHA or
            saved["dual_sha"]!=BASE_DUAL_SHA or
            saved["token_sha"]!=provenance["token_sha256"]):
            raise ValueError("Unsafe resumed optimizer checkpoint provenance")
        for key,q in sources.items():
            q.centroids.data.copy_(saved["centroids"][key].to(device))
        later.load_state_dict(saved["later"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.state.values():
            for k,v in group.items():
                if isinstance(v,torch.Tensor):group[k]=v.to(device)
        start=int(saved["step"])
    print(json.dumps({"event":"CALIBRI_S59_PHYSICAL_TRAIN_START",
      "device":str(device),"steps":a.steps,"resume_from":start,
      "source_bytes":len(source),"base_bpp":8*(len(source)+len(dual)+len(manifest))/SOURCE_PARAMS,
      "text_train_provenance":provenance,
      "learning_rates":{"centroids":a.centroid_lr,"tunnel":a.tunnel_lr},
      "loss_weights":{"LM":a.lm_weight,"historical":a.old_replay_weight},
      "CPU_CONTINUATION_REQUIRES_NO_GPU":True}),flush=True)
    timer=time.monotonic()
    log=[]
    for step in range(start+1,start+a.steps+1):
        optimizer.zero_grad(set_to_none=True)
        # All sample indices fixed by step number, so resuming on CPU changes
        # only the compute backend (rounding differences possible, recorded).
        k=(step*31+SEED)%a.windows
        start_idx=k*a.context
        window=ids[start_idx:start_idx+a.context].to(device)
        pair_a=original[(step*7)%len(original)]
        pair_b=augmented[(step*11)%len(augmented)]
        oldpair=old[(step*5)%4]
        # Always train BOTH factual sides of the same contrasting noun
        # pair, not a single parity-dependent side. This prevents
        # appearance of correct single answers from unlearned pair binding.
        lm=text_loss(model,sources,later,window,device)
        firstloss=.5*(example_loss(model,sources,first,later,pair_a["left"],device)+
                      example_loss(model,sources,first,later,pair_a["right"],device))
        secondloss=.5*(example_loss(model,sources,first,later,pair_b["left"],device)+
                       example_loss(model,sources,first,later,pair_b["right"],device))
        historic=.5*(example_loss(model,sources,first,later,oldpair["left"],device)+
                     example_loss(model,sources,first,later,oldpair["right"],device))
        total=a.lm_weight*lm+firstloss+secondloss+a.old_replay_weight*historic
        if not torch.isfinite(total):raise FloatingPointError("Non-finite TRAIN update")
        total.backward()
        params=paid_centroids+list(later.parameters())
        torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        optimizer.step()
        clear(sources)
        row={"step":step,"loss_for_optimizer_only":float(total.detach().cpu()),
             "device":str(device)}
        log.append(row)
        if step%a.checkpoint_every==0 or step==start+a.steps:
            packed,diag=replace_discrete(archive,sources,strict=False)
            physical_path=a.out/f"S59_STEP_{step:04d}_PAID_RESEARCH.cb59"
            physical_path.write_bytes(packed)
            f=a.out/f"S59_STEP_{step:04d}_TRAIN_RESUME.pt"
            cpu_opt=optimizer.state_dict()
            torch.save({"step":step,"source_sha":BASE_SOURCE_SHA,
              "dual_sha":BASE_DUAL_SHA,"token_sha":provenance["token_sha256"],
              "centroids":{k:q.centroids.detach().cpu() for k,q in sources.items()},
              "later":{k:v.detach().cpu() for k,v in later.state_dict().items()},
              "optimizer":cpu_opt},f)
            side=a.out/f"S59_STEP_{step:04d}_GREEN_TUNNEL.safetensors"
            tensors={"first."+k:v.detach().cpu().contiguous() for k,v in first.state_dict().items()}
            tensors.update({"later."+k:v.detach().cpu().contiguous() for k,v in later.state_dict().items()})
            save_file(tensors,str(side))
            phys=paid_manifest(packed,side.read_bytes(),a.out/f"S59_STEP_{step:04d}_PAID_MANIFEST.json")
            row.update({"changed_FP16_dictionary_cells":diag["actually_changed_FP16_codebook_cells"],
                        "physical_bpp":phys["all_inclusive_bpp"],
                        "total_train_time_seconds":round(time.monotonic()-timer,3)})
            print(json.dumps({"event":"CALIBRI_S59_CHECKPOINT","status":"candidate_only",**row}),flush=True)
    # Full native accuracy must be measured on CPU against IMMUTABLE S56.
    # This is the only promotion gate, never metric-only LM loss or QA peeking.
    final=log[-1]["step"]
    packed=(a.out/f"S59_STEP_{final:04d}_PAID_RESEARCH.cb59").read_bytes()
    newdual=(a.out/f"S59_STEP_{final:04d}_GREEN_TUNNEL.safetensors").read_bytes()
    baseline=probe_native_cpu(source,dual,tok,original,augmented,old)
    candidate=probe_native_cpu(packed,newdual,tok,original,augmented,old)
    b0,b1,b2=[quality(r,p) for r,p in zip(baseline,(original,augmented,old[:4]))]
    n0,n1,n2=[quality(r,p) for r,p in zip(candidate,(original,augmented,old[:4]))]
    assert len(b0[0])==42 and len(b1[0])==27 and len(b2[0])==7
    assert len(b0[1])==58 and len(b1[1])==46
    assert b0[2]==13 and b1[2]==7
    old_retained=(b0[0]<=n0[0] and b1[0]<=n1[0] and b2[0]<=n2[0] and
                  b0[1]<=n0[1] and b1[1]<=n1[1])
    both_more=(n0[2]+n1[2])>(b0[2]+b1[2])
    train_guard_pass=bool(old_retained and both_more)
    candidate_accepted=False  # Never promote on TRAIN. Require unseen QA + CPU speed.
    report={"stage":"CALIBRI_S59_PHYSICAL_CENTROID_AND_GREEN_TUNNEL_GLOBAL_TRAIN",
      "device_used":str(device),"train_steps":a.steps,"last_step":final,
      "source_sha":sha(packed),"result":{ "old_skills_preserved":old_retained,
          "both_pairs_improved":both_more,
          "candidate_eligible_on_train_only":train_guard_pass,
          "promote_on_train":False,
          "requires_precommitted_independent_unseen_QA_and_CPU_speed_check":True,
          "baseline":{"original_full":len(b0[0]),"added_full":len(b1[0]),"old_full":len(b2[0]),
                      "original_pairs":b0[2],"added_pairs":b1[2]},
          "candidate":{"original_full":len(n0[0]),"added_full":len(n1[0]),"old_full":len(n2[0]),
                      "original_pairs":n0[2],"added_pairs":n1[2]}},
      "all_steps":log,"CPU_resume_stored_even_if_candidate_rejected":True,
      "not_independent_QA_or_FP32_parity":True,
      "no_S29_S26_S53_S55_S57_used":True,
      "no_LIVE_Jessica_Render_Supabase_APK_changes":True}
    (a.out/"S59_TRAIN_ONLY_COMPLETE_REPORT.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"event":"CALIBRI_S59_RESULT","accepted":False,
                       "train_guard_pass":train_guard_pass,
                       "metrics":report["result"]}),flush=True)

if __name__=="__main__":
    main()
