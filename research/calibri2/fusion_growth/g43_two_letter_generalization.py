"""G43: incremental two-letter generalization from exact G42 Balanced.
Replay all eight G42 TRAIN prompts, expand to 16 then 32.
Same existing native head and parallel slot; all other weights frozen.
Separate old DEV and preregistered new paired wording tests; no DEV selection.
"""
import argparse,json,os,time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load,save
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import Residual,install,parameters,PIN,restore,digest,G05,block
from research.calibri2.fusion_growth.g36_alphabet_margin import encode,evaluate
from research.calibri2.fusion_growth.g41_balanced_joint import Balance,balance_scope
from v176.operators import clear

PARENT={"balanced":"d3b5ffc9b60614598a7bf9b76f932894a8af6fa124499d0fe0f6e749ee8f9012"}
LETTERS=("а","м")
TRAIN=("Напиши букву {}.","Повтори букву «{}». Напиши только букву.","Скопируй символ {}.","Ответь одной буквой: {}.",
"{} — повтори эту букву.","Выведи строчную букву {} без пояснений.","Нужен один символ: {}.","Запиши ровно тот знак, который указан здесь: {}.",
"Пожалуйста, повтори {}.","В ответе оставь только «{}».","{}\nПерепиши этот знак.","Напечатай следующую букву — {}.",
"Символ для копирования: {}. Не добавляй других знаков.","Покажи букву {} в ответе.","Напиши мне один знак ({}) без скобок.","Требуется воспроизвести букву [{}].")
DEV=("Воспроизведи написанный символ «{}».","Какую букву я написал: {}?","Укажи символ между скобками ({}) .","Напечатай только знак: {}!")

FRESH=("Верни единственный символ «{}».","{} — это буква для ответа.","Запиши без кавычек: «{}».","Перенеси знак {} в ответ без изменений.","Мне нужна именно эта буква: {}.","Укажи только содержимое квадратных скобок [{}].","Воспроизведи знак после двоеточия: {}","Ответ должен состоять из символа {}.","Сейчас напиши {} и больше ничего.","Какой символ следует перепечатать? {}","Буква для записи — {}. Выполни задание.","Сделай копию указанного знака: {}.")
def capture(st,r,k,slot):
    model=st[0];h=[]
    hook=model.lm_head.register_forward_pre_hook(lambda m,a:h.append(a[0][:,-1:,:].detach().clone()))
    try:z,_=block(st,r["p"]+r["g"][:k],r["features"],slot)
    finally:hook.remove()
    assert len(h)==2 and h[0].shape==(1,1,320)
    return h[0][0,0],z.detach()[0],r["g"][k:k+2]

def pack(cache):
    h=torch.stack([r[0] for r in cache]).to("mps")
    z=torch.stack([r[1] for r in cache]).to("mps")
    y=torch.full((len(cache),2),-100,dtype=torch.long)
    for i,r in enumerate(cache):y[i,:len(r[2])]=torch.tensor(r[2])
    return h,z,y.to("mps")

def run(g05,f08,parent,out,arm):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.manual_seed(20270129);torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train=[encode(tok,t.format(a),a) for t in TRAIN for a in LETTERS]
    dev=[encode(tok,t.format(a),a) for t in DEV for a in LETTERS]
    fresh=[encode(tok,t.format(a),a) for t in FRESH for a in LETTERS]
    assert len(train)==32 and len(dev)==8 and len(fresh)==24
    assert len({r["q"] for r in train+dev+fresh})==64
    (out/"SPLITS.json").write_text(json.dumps({k:[{"q":r["q"],"answer":r["answer"],"g":r["g"]} for r in v] for k,v in (("TRAIN",train),("OLD_DEV",dev),("FRESH",fresh))},ensure_ascii=False,indent=2))
    print("G43_TARGETS",json.dumps([{"q":r["q"],"g":r["g"]} for r in train[:2]],ensure_ascii=False),flush=True)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    model,sources,_,full,_,child,_=st
    packet=(parent/"two_letter_joint.safetensors").read_bytes();assert digest(packet)==PARENT[arm]
    raw=load(packet);l6=Residual();slot=Residual();balance=Balance()
    for prefix,mod in (("l6.",l6),("slot.",slot),("head.",child.language_head)):
        mod.load_state_dict({k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix)},strict=True)
    with torch.no_grad():
        for k,v in raw.items():
            if k.startswith("core."):sources[k[5:]].centroids.copy_(v.float())
            elif k=="GT.gates":full.gates.copy_(v.float())
            elif k=="balance.logits":balance.logits.copy_(v.float())
        weight=model.lm_head.source().detach().clone().float()
        clear(sources)
    assert weight.shape==(49152,320)
    with install(st,6,l6),balance_scope(st,l6,balance,arm):
        before,records=evaluate(st,tok,train[:8],slot)
        base_dev,rd=evaluate(st,tok,dev,slot)
        base_fresh,rf=evaluate(st,tok,fresh,slot)
        assert all(v["exact_EOS"]==8 for v in before.values()),before
        (out/"BASELINE.json").write_text(json.dumps({"replay":records,"old_dev":rd,"fresh":rf},ensure_ascii=False))
        cached=[[capture(st,r,k,slot) for k in range(len(r["g"]))] for r in train]
    # Only existing head and slot are moved/trained. Native core is fixed.
    head=child.language_head.to("mps");slot.to("mps");w=weight.to("mps")
    ps=list(head.parameters())+list(slot.parameters())
    for p in ps:p.requires_grad_(True)
    frozen=list({id(p):p for p in list(parameters(restored))+list(l6.parameters())+list(balance.parameters()) if id(p) not in {id(v) for v in ps}}.values())
    versions={id(p):p._version for p in frozen}
    def logits(h):
        h=h[:,None,:];s=slot(h)
        return torch.cat((F.linear(h,w)+.25*head(h),F.linear(s,w)+.25*head(s)),dim=1).float()
    h,z,y=pack([x for rr in cached for x in rr])
    with torch.no_grad():
        pred=logits(h)
        err=float((pred-z).abs().max())
        same=bool(torch.equal(pred.argmax(-1),z.argmax(-1)))
    print("G43_CACHE_PARITY",json.dumps({"max_abs_error":err,"top1_equal":same}),flush=True)
    assert err<.002 and same,(err,same)
    opt=torch.optim.AdamW(ps,lr=.005,foreach=False,weight_decay=0.)
    curve=[];stage_reports=[];start=time.perf_counter();updates=0;final=packet
    def checkpoint():
        fields=dict(raw)
        for prefix,mod in (("head.",head),("slot.",slot)):
            for k,v in mod.state_dict().items():fields[prefix+k]=v.detach().cpu().half().contiguous()
        return save(fields)
    def measure(h,y):
        with torch.no_grad():
            p=logits(h);ok=y!=-100
            v=p[ok];t=y[ok];gold=v.gather(1,t[:,None]).squeeze(1)
            wrong=v.clone();wrong.scatter_(1,t[:,None],-torch.inf)
            return {"correct":int((v.argmax(1)==t).sum()),"n":len(t),"min_margin":float((gold-wrong.max(1).values).min())}
    for stage,n,cap in (("sixteen_examples",16,1000),("thirtytwo_examples",32,1600)):
        h,z,y=pack([x for rr in cached[:n] for x in rr])
        reached=False
        for step in range(1,cap+1):
            p=logits(h);ok=y!=-100;v=p[ok];t=y[ok]
            gold=v.gather(1,t[:,None]).squeeze(1)
            wrong=v.clone();wrong.scatter_(1,t[:,None],-torch.inf)
            # Equal supervision of letter bytes/tokens and EOS, both slots.
            loss=F.cross_entropy(v,t)+F.softplus(2.+wrong.max(1).values-gold).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            for q in ps:
                assert q.grad is not None and q.grad.device.type=="mps" and torch.isfinite(q.grad).all()
            torch.nn.utils.clip_grad_norm_(ps,10.,error_if_nonfinite=True);opt.step();updates+=1
            if step==1 or step%10==0 or step==cap:
                m=measure(h,y)
                item={"stage":stage,"step":step,"updates":updates,"loss":float(loss.detach()),**m}
                if step==1 or step%50==0 or m["min_margin"]>=2.:
                    curve.append(item);print("G43_PROGRESS",json.dumps(item),flush=True)
                if m["correct"]==m["n"] and m["min_margin"]>=2.:
                    reached=True;break
        final=checkpoint();(out/(stage+".safetensors")).write_bytes(final)
        # Strict FP16 reload of trained weights before checking gate.
        fields=load(final)
        for prefix,mod in (("head.",head),("slot.",slot)):
            mod.load_state_dict({k[len(prefix):]:v for k,v in fields.items() if k.startswith(prefix)},strict=True)
        m=measure(h,y);passed=m["correct"]==m["n"]
        stage_reports.append({"stage":stage,"steps":step,"margin_gate":reached,"reload":m,"sha":digest(final),"passed":passed})
        print("G43_STAGE",json.dumps(stage_reports[-1]),flush=True)
        if not passed:break
    head.to("cpu");slot.to("cpu")
    assert all(p._version==versions[id(p)] for p in frozen)
    with install(st,6,l6),balance_scope(st,l6,balance,arm):
        exact,records=evaluate(st,tok,train[:8],slot)
        expanded,more=evaluate(st,tok,train,slot)
        new,extra=evaluate(st,tok,dev,slot)
        fresh_score,fs=evaluate(st,tok,fresh,slot)
    (out/"TRAIN_AND_DEV.json").write_text(json.dumps({"replay":records,"expanded":more,"old_dev":extra,"fresh":fs},ensure_ascii=False))
    (out/"g43_two_letter_joint.safetensors").write_bytes(final)
    size=sum(map(len,blobs))+sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())+len(final)
    bpp=8*size/G05;assert bpp<=.7
    report={"arm":arm,"parent_sha":PARENT[arm],"checkpoint_sha":digest(final),"letters":LETTERS,"baseline_replay":before,"baseline_old_dev":base_dev,"baseline_fresh":base_fresh,"replay":exact,"expanded_train":expanded,"old_dev":new,"fresh":fresh_score,"stages":stage_reports,"curve":curve,"updates":updates,"training_and_evaluation_seconds":time.perf_counter()-start,"known_payload_bpp":bpp,"replay_gate":all(v["exact_EOS"]==8 for v in exact.values()),"expanded_gate":all(v["exact_EOS"]==32 for v in expanded.values()),"candidate_accepted":False,"limitations":["Frozen actual G42 Balanced core, GT, NRA3 and L6; existing head and slot trained on cached real CPU features on MPS, not full-core GPU backprop. No added neural regions. Same native full vocabulary, no inference masks.","Two fixed letters, memorization gate not alphabet or general language competence. Old DEV not used for G43 updates; FRESH 24 new authored prompts fixed before training, exploratory not sealed blind. No DEV-based checkpoint selection. No 616 or CPU speed certification."]}
    (out/"G43_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2));print("G43_REPORT",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--arm",choices=("balanced",),required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.out,a.arm)
