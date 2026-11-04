"""Learn a narrow *candidate critic* on independent gold answers, not logits.

No examples from validity probes enter training. This does not magically
restore missing facts or replace an independently qualified language model.
"""
import hashlib,json,random,time
import torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from .protocol import SEED,OPTIM_STEPS,BATCH_SIZE,LR,MAX_CHECKPOINT_BYTES,VALID_A,VALID_B
from .critic import GreenTunnel
from .examples import make_train_pairs,validation_pairs
from .bytes_adapter import pairs

def infer(model,examples):
    model.eval()
    with torch.inference_mode():
        logits=[]
        for start in range(0,len(examples),24):
            rows=[(q,a) for q,a,_ in examples[start:start+24]]
            logits.extend(torch.sigmoid(model(pairs(rows))).tolist())
    return logits

def accuracy(records,predictions):
    yes=[s for (_,_,label),s in zip(records,predictions) if label==1]
    no=[s for (_,_,label),s in zip(records,predictions) if label==0]
    return {"positive_recall":sum(x>=.5 for x in yes)/len(yes),
            "negative_rejection":sum(x<.5 for x in no)/len(no),
            "mean_positive_probability":sum(yes)/len(yes),
            "mean_negative_probability":sum(no)/len(no)}

def fit(out):
    random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(2)
    net=GreenTunnel()
    n=net.disk_parameters()
    if n*4>=MAX_CHECKPOINT_BYTES:
        raise AssertionError("not a <10MB independent FP32 neural device")
    rows=make_train_pairs()
    source=pairs([(q,a) for q,a,y in rows])
    labels=torch.tensor([float(y) for _,_,y in rows])
    opt=torch.optim.Adam(net.parameters(),lr=LR)
    rng=random.Random(SEED)
    curve=[]
    started=time.perf_counter()
    for step in range(OPTIM_STEPS):
        idx=[rng.randrange(len(rows)) for _ in range(BATCH_SIZE)]
        pred=net(source[idx])
        b=labels[idx]
        loss=F.binary_cross_entropy_with_logits(pred,b,pos_weight=torch.tensor(4.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step()
        if step%20==0 or step==OPTIM_STEPS-1:
            curve.append([step,round(float(loss.detach()),5)])
    path=out/"GREEN_TUNNEL_critic.safetensors"
    save_file({k:v.detach().cpu().contiguous()
               for k,v in net.state_dict().items()},str(path))
    raw=path.read_bytes()
    if len(raw)>MAX_CHECKPOINT_BYTES:
        raise AssertionError("physical 10MB limit breached including metadata")
    restored=GreenTunnel()
    restored.load_state_dict(load_file(str(path)),strict=True)
    controls=validation_pairs(VALID_A+VALID_B)
    valid_prob=infer(restored,controls)
    # Reload must leave all probability bits unchanged on CPU.
    from_mem=infer(net,controls)
    if any(abs(a-b)>1e-7 for a,b in zip(valid_prob,from_mem)):
        raise AssertionError("model physical reload changed critic verdict")
    return restored,{
        "network_type":"4-layer self-contained byte-level neural candidate critic",
        "unique_neural_parameters":n,
        "full_weights_and_metadata_file_bytes":len(raw),
        "checkpoint_sha256":hashlib.sha256(raw).hexdigest(),
        "under_10_000_000_bytes":True,
        "train_source_questions":len(set(q for q,_,_ in rows)),
        "train_examples_with_corruptions":len(rows),
        "fit_curve_binary_crossentropy_supervised":curve,
        "training_wall_seconds":time.perf_counter()-started,
        "gold_answer_ranking_not_generation":True,
        "heldout_never_optimized":True,
        "heldout_gold_vs_synthetic_corrupt_metrics":accuracy(controls,valid_prob),
        "does_not_verify_arbitrary_unseen_world_facts":True,
        "does_not_include_original_host_llm_storage":True,
    }
