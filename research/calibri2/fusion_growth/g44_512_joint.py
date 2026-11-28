"""G44: owner-approved 512 unique two-letter TRAIN prompts from exact G43.
Full MPS joint core/GT/L6/head/slot with low LR and protected TRAIN replay.
Frozen NRA3 and learned residual budgets; no architectural additions.
Full native vocabulary; no gold input at inference; fixed fresh evaluation.
"""
import argparse,json,os,random,time
from types import MethodType
from contextlib import contextmanager
from torch import nn
from research.calibri2.fusion_growth.g35_full_mps_preflight import native,native_device_mask
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load,save
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import Residual,install,parameters,PIN,restore,digest,forward,G05,block
from research.calibri2.fusion_growth.g28_l6_continuation import ByteValidity
from research.calibri2.fusion_growth.g43_two_letter_generalization import TRAIN as OLD_TRAIN, DEV as OLD_DEV, FRESH as OLD_FRESH
LETTERS=("а","м")
from research.calibri2.fusion_growth.f21_native_cpu_integration import prompt_features,strict_chat_source_ids
from v176.operators import clear
PARENT={"balanced":"2dc970386a5aa761d184d770e83998abe60f7beb288e3986f03bf99549850556"}
SEED=20270130
EPOCHS=1
BATCH=16

def encode(tok,q,a):
    p=strict_chat_source_ids(tok,q)
    chat=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    ids=tok.encode(chat+a,add_special_tokens=False);g=ids[len(p):]+[2]
    assert ids[:len(p)]==p and 2 not in g[:-1] and len(p)+len(g)<160 and tok.decode(g[:-1],skip_special_tokens=True).strip()==a
    return {"q":q,"answer":a,"p":p,"g":g,"features":prompt_features(tok,q)}

# 40 authored instruction frames x six source-marker styles x two letters,
# plus exactly 32 inherited TRAIN. 240 paired combinations, not 240 independent
# semantic tasks; both target letters occur in every template.
FRAMES=(
"Перепиши {} в ответ.","Напечатай букву {} строчным шрифтом.","Введи знак {} и закончи ответ.",
"Мне нужен символ {} без лишних слов.","Оставь в сообщении лишь {}.","Пожалуйста, скопируй этот символ: {}.",
"Сделай ответ из одной буквы {}.","{} — напечатай этот знак.","{} — перепиши без изменений.",
"Текст для копирования: {}. Верни его.","Исходный знак: {}. Запиши его в ответе.",
"Выполни короткое задание: повтори {}.","От тебя требуется одна буква — {}.",
"Поставь в ответе символ {}.","Набери с клавиатуры букву {}.","Впиши {} вместо ответа.",
"Воспроизведи ровно один знак: {}.","Укажи строчную букву {}.","Сохрани исходный символ {} в ответе.",
"Перенеси {} в своё сообщение.","Напечатай {} без комментариев.","{} — это образец. Скопируй его.",
"Изобрази букву {} текстом.","Выдай в ответе только букву {}.","Запиши указанный знак {} без дополнительных символов.",
"Ничего не объясняй, просто напиши {}.","Ответь текстом из одного символа: {}.",
"Я указал букву {}. Повтори её.","Для ответа используй единственный знак {}.",
"Перепечатай этот образец: {}.","Нужно повторить строчную букву {}.",
"Следующая буква предназначена для копирования: {}.","{} — знак, который нужно перенести в ответ.",
"Пусть ответом будет только {}.","Скопируй в точности букву {}.",
"Вместо пояснения пришли знак {}.","Задание на запись символа: {}.",
"Вот нужная буква: {}. Напиши её.","Напечатай указанную букву {} и остановись.",
"Продублируй один символ: {}."
)
STYLES=("{}", "«{}»", "({})", "[{}]", "'{}'", "【{}】")
# Distinct authored instruction frames, not punctuation variants of TRAIN.
NEW_DEV=(
"Какой знак указан в конце этой просьбы? {}",
"Сделай буквальный ответ из содержимого поля: {}",
"На карточке стоит буква {}. Перенеси её сюда.",
"Мне не нужен её смысл — пришли саму букву {}.",
"Как выглядит указанный здесь символ? {}",
"Проверь, что в ответе есть ровно один знак: {}",
"В этой задаче ответ заранее дан: {}. Введи его.",
"Мой образец состоит из буквы {}. Напечатай образец.",
"Покажи, что умеешь повторять одиночные знаки. Образец: {}.",
"Пришли копию буквы, которую я указал: {}.",
"В поле ввода должна стоять {}. Что ты туда введёшь?",
"Возьми следующий одиночный знак и верни его: {}.",
"Тебе дан символ {}. Ответ должен совпасть с ним.",
"Какой символ написан после слова образец? Образец {}.",
"Здесь записан один знак — {}. Какой именно?",
"Замени всю фразу в своём ответе одной буквой {}.",
"Мне требуется точная копия этого текста: {}.",
"Только содержимое поля «{}» должно появиться в ответе.",
"Перепиши один знак из строки ниже:\n{}",
"Обозначение для ввода: {}. Не нужно его расшифровывать.",
"Считай следующий знак готовым ответом: {}.",
"Воспроизведи символ, стоящий перед точкой: {}.",
"Дай ответ, совпадающий с этим обозначением: {}.",
"Какую букву следует вывести, если образец — {}?",
"Не меняя регистр, верни символ «{}».",
"Заполни ответ тем же знаком, который видишь здесь: {}.",
"Единственное содержимое ответа задано: {}.",
"Напечатай символ, заключённый между вертикальными чертами |{}|.",
"Представь, что копируешь одну ячейку таблицы: {}.",
"Что написано в поле «буква»: {}?",
"Ответь не названием буквы, а самим знаком: {}.",
"Одиночный символ для переноса в сообщение — {}."
)
def suite(tok):
    train=[encode(tok,t.format(a),a) for t in OLD_TRAIN for a in LETTERS]
    train += [encode(tok,t.format(s.format(a)),a) for t in FRAMES for s in STYLES for a in LETTERS]
    old=[encode(tok,t.format(a),a) for t in OLD_DEV+OLD_FRESH for a in LETTERS]
    fresh=[encode(tok,t.format(a),a) for t in NEW_DEV for a in LETTERS]
    assert len(train)==512 and len(old)==32 and len(fresh)==64
    assert len({r["q"] for r in train+old+fresh})==608
    return train,old,fresh

def evaluate(st,tok,rows,slot):
    totals={};records=[]
    for mode in ("sequential","parallel"):
        d={"n":len(rows),"exact_EOS":0,"correct_text":0,"EOS":0,"invalid":0,"empty":0,"first_correct":0,"overrun":0}
        for r in rows:
            ids=[];cap=8
            while len(ids)<cap:
                if mode=="sequential":
                    z=forward(st,r["p"]+ids,r["features"]).detach();ts=[int(z.argmax())]
                else:
                    z,_=block(st,r["p"]+ids,r["features"],slot);ts=z.detach().argmax(-1)[0].tolist()
                for t in ts:
                    if len(ids)>=cap:break
                    ids.append(t)
                    if t==2:break
                if ids[-1]==2:break
            a=tok.decode(ids,skip_special_tokens=True).strip()
            for k,v in {"exact_EOS":a==r["answer"] and ids[-1]==2,"correct_text":a==r["answer"],"EOS":ids[-1]==2,"invalid":"\ufffd" in a,"empty":not a,"first_correct":ids[0]==r["g"][0],"overrun":len(ids)==cap and ids[-1]!=2}.items():d[k]+=int(v)
            records.append({"mode":mode,"q":r["q"],"expected":r["answer"],"answer":a,"ids":ids})
        totals[mode]=d
    return totals,records


class Balance(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=nn.Parameter(torch.zeros(2))
    def budgets(self):
        return .5*self.logits.sigmoid()
    def bounded(self,h,delta,i):
        # Smooth cap: ||applied delta|| <= budget * ||h|| per token.
        limit=self.budgets()[i]*h.float().norm(dim=-1,keepdim=True).clamp_min(1.e-6)
        norm=delta.float().norm(dim=-1,keepdim=True)
        return delta*(limit/(limit+norm+1.e-6)).to(delta.dtype)

@contextmanager
def balance_scope(st,l6,balance,arm):
    model,_,_,_,_,child,_=st
    seen={}
    def at5(m,a,o):
        h=o[0] if isinstance(o,tuple) else o
        seen["h5"]=h[:,-1,:]
    def at6(m,a,o):
        h=a[0]
        seen["pre6"]=h[:,-1,:]
        result=h+balance.bounded(h,o-h,0) if arm=="balanced" else o
        seen["post6"]=result[:,-1,:]
        return result
    def atchild(m,a,o):
        h=seen["post6"]
        assert h.shape==o.shape
        return balance.bounded(h,o,1) if arm=="balanced" else o
    def at7(m,a):
        seen["h7"]=a[0][:,-1,:]
    hooks=[model.model.layers[4].register_forward_hook(at5),l6.register_forward_hook(at6),child.register_forward_hook(atchild),model.model.layers[6].register_forward_pre_hook(at7)]
    try:yield seen
    finally:
        for h in hooks:h.remove()

def preservation(seen,previous):
    # Paired TRAIN-only same-template different-letter prompts.
    # Match at least half of the L5 normalized distance after injection.
    h5=F.normalize(seen["h5"].float(),dim=-1)
    h7=F.normalize(seen["h7"].float(),dim=-1)
    if previous is None:loss=h7.sum()*0.
    else:
        target=.5*(h5.detach()-previous[0]).square().sum(-1)
        actual=(h7-previous[1]).square().sum(-1)
        loss=(F.relu(target-actual)/(target+1.e-4)).mean()
    return loss,(h5.detach(),h7.detach())

def run(g05,f08,parent,out,arm):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.set_num_threads(2);torch.manual_seed(SEED)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train,dev,fresh=suite(tok)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    model,sources,bridge,full,expansion,child,_=st
    out.mkdir(parents=True,exist_ok=False)
    assert arm=="balanced"
    balance=Balance()
    packet=(parent/"g43_two_letter_joint.safetensors").read_bytes();assert digest(packet)==PARENT[arm]
    raw=load(packet);l6=Residual();l6.load_state_dict({k[3:]:v for k,v in raw.items() if k.startswith("l6.")},strict=True)
    slot=Residual();slot.load_state_dict({k[5:]:v for k,v in raw.items() if k.startswith("slot.")},strict=True)
    child.language_head.load_state_dict({k[5:]:v for k,v in raw.items() if k.startswith("head.")},strict=True)
    # Explicit compact existing codebooks only; no dense trainable copy.
    keys=[f"model.layers.{i}.{k}.weight" for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    assert all(k in sources for k in keys)
    selected={"head."+k:p for k,p in child.language_head.named_parameters()}|{"slot."+k:p for k,p in slot.named_parameters()}
    if arm=="balanced":
        selected|={"l6."+k:p for k,p in l6.named_parameters()}
        selected|={"core."+k:sources[k].centroids for k in keys}
        selected["GT.gates"]=full.gates
    with torch.no_grad():
        for k,p in selected.items():
            assert k in raw and raw[k].shape==p.shape
            p.copy_(raw[k].float())
    with torch.no_grad():balance.logits.copy_(raw["balance.logits"].float())
    (out/"SPLITS.json").write_text(json.dumps({k:[{"q":r["q"],"answer":r["answer"],"g":r["g"]} for r in v] for k,v in (("TRAIN",train),("OLD_DEV",dev),("FRESH",fresh))},ensure_ascii=False,indent=2))
    # Exact CPU compatibility with the parent balance hooks active.
    with install(st,6,l6),balance_scope(st,l6,balance,arm):
        cpu_ref=[z.detach().clone() for z in native(st,train[0]["p"],train[0]["features"],slot)]
    mid=child.child.child
    mid.delta_components=MethodType(native_device_mask,mid)
    with install(st,6,l6),balance_scope(st,l6,balance,arm):
        cpu_patched=[z.detach() for z in native(st,train[0]["p"],train[0]["features"],slot)]
    assert all(torch.equal(x,y) for x,y in zip(cpu_ref,cpu_patched))
    with install(st,6,l6),balance_scope(st,l6,balance,arm):
        before,items=evaluate(st,tok,train[:32],slot)
        assert all(d["exact_EOS"]==32 for d in before.values()),before
        old_before,old_items=evaluate(st,tok,dev,slot)
        fresh_before,fresh_items=evaluate(st,tok,fresh,slot)
    (out/"PARENT_DEV.json").write_text(json.dumps({"replay":items,"old":old_items,"fresh":fresh_items},ensure_ascii=False))
    print("G44_PARENT",arm,json.dumps(before),flush=True)
    # Off-model GT/bridge/expansion modules and weak-referenced original donor
    # also need movement; module .to alone would miss some research hooks.
    donor=bridge._existing()
    modules=[model,bridge,full,expansion,child,l6,slot,balance]
    if donor is not None:modules.append(donor)
    for bank in restored[1:]:
        if isinstance(bank,dict):modules.extend(v for v in bank.values() if isinstance(v,torch.nn.Module))
        elif isinstance(bank,torch.nn.Module):modules.append(bank)
    # Legacy donor can be alive only through hook defaults, not bridge weakref.
    # Include the exact bound module objects from installed hooks.
    for owner in list(model.modules()):
        for hook in owner._forward_hooks.values():
            for obj in list(getattr(hook,"__defaults__",None) or ()) + list((getattr(hook,"__kwdefaults__",None) or {}).values()):
                if isinstance(obj,torch.nn.Module):modules.append(obj)
            for cell in getattr(hook,"__closure__",None) or ():
                obj=cell.cell_contents
                if isinstance(obj,torch.nn.Module):modules.append(obj)
            bound=getattr(hook,"__self__",None)
            if isinstance(bound,torch.nn.Module):modules.append(bound)
    modules=list({id(m):m for m in modules}.values())
    for m in modules:m.to("mps")
    (out/"DEVICE_MODULES.json").write_text(json.dumps([type(m).__name__ for m in modules]))
    for s in sources.values():s.clear();s.to("mps")
    assert all(p.device.type=="mps" for m in modules for p in m.parameters())
    assert all(b.device.type=="mps" for m in modules for b in m.buffers())

    # Resolve selected references again after module.to, before optimizer construction.
    selected={"head."+k:p for k,p in child.language_head.named_parameters()}|{"slot."+k:p for k,p in slot.named_parameters()}|{"l6."+k:p for k,p in l6.named_parameters()}|{"core."+k:sources[k].centroids for k in keys}|{"GT.gates":full.gates}
    assert all(p.device.type=="mps" for p in selected.values())
    for p in selected.values():p.requires_grad_(True)
    with install(st,6,l6),balance_scope(st,l6,balance,arm):
        gpu_ref=native(st,train[0]["p"],train[0]["features"],slot)
        err=max(float((x.detach().cpu()-y).abs().max()) for x,y in zip(gpu_ref,cpu_ref))
        print("G44_MPS_PARITY",json.dumps({"max_abs_error":err,"top1_cpu":[int(z.argmax()) for z in cpu_ref],"top1_gpu":[int(z.argmax()) for z in gpu_ref]}),flush=True)
        assert err<.002 and all(int(x.argmax())==int(y.argmax()) for x,y in zip(gpu_ref,cpu_ref))
        assert all(torch.isfinite(z).all() for z in gpu_ref)
        del gpu_ref
    start=time.perf_counter()
    frozen=list({id(p):p for p in list(parameters(restored))+list(l6.parameters())+list(slot.parameters())+list(balance.parameters()) if id(p) not in {id(v) for v in selected.values()}}.values())
    versions={id(p):p._version for p in frozen}
    for p in frozen:p.grad=None
    lr=lambda k:.0002 if k.startswith(("head.","slot.")) else .00003
    opt=torch.optim.AdamW([{"params":[p],"lr":lr(k)} for k,p in selected.items()],foreach=False,weight_decay=0.)
    groups={k:("core" if k.startswith("core.") else k.split(".")[0]) for k in selected}
    init={k:p.detach().clone() for k,p in selected.items()}
    rng=random.Random(SEED);curve=[];seen=set();steps=0
    def letter_loss(z,t):
        # Same equal-weight full vocabulary CE + top1 margin as G43.
        gold=z[0,t];wrong=z[0].clone();wrong[t]=-torch.inf
        return F.cross_entropy(z,torch.tensor([t],device=z.device))+F.softplus(2.+wrong.max()-gold)
    def serialize():
        payload=dict(raw)
        payload.update({k:p.detach().cpu().half().contiguous() for k,p in selected.items()})
        for prefix,mod in (("head.",child.language_head),("slot.",slot),("l6.",l6)):
            for k,p in mod.state_dict().items():payload[prefix+k]=p.detach().cpu().half().contiguous()
        return save(payload)
    with install(st,6,l6),balance_scope(st,l6,balance,arm) as states:
        for ep in range(1,EPOCHS+1):
            pairs=list(range(256));rng.shuffle(pairs)
            order=[2*i+j for i in pairs for j in (0,1)]
            total=0.;tokens=0;visited=set()
            for j in range(0,len(order),BATCH):
                base=order[j:j+BATCH];visited.update(base)
                pair=(j//BATCH)%16
                batch=base+[2*pair,2*pair+1];opt.zero_grad(set_to_none=True)
                acc={k:torch.zeros_like(p) for k,p in selected.items()}
                for i in batch:
                    r=train[i];g=r["g"]
                    for pos,t in enumerate(g):
                        z,z2=native(st,r["p"]+g[:pos],r["features"],slot)
                        assert z.device.type==z2.device.type=="mps"
                        loss=letter_loss(z,t)
                        if pos+1<len(g):loss=loss+letter_loss(z2,g[pos+1])
                        total+=float(loss.detach());tokens+=1
                        grads=torch.autograd.grad(loss/(len(batch)*len(g)),tuple(selected.values()),allow_unused=True)
                        for (k,p),grad in zip(selected.items(),grads):
                            if grad is not None:
                                assert grad.device.type=="mps" and torch.isfinite(grad).all()
                                acc[k]+=grad.detach()
                                if float(grad.abs().sum())>0:seen.add(groups[k])
                for k,p in selected.items():p.grad=acc[k];assert p.grad.device.type=="mps"
                torch.nn.utils.clip_grad_norm_(list(selected.values()),1.,error_if_nonfinite=True)
                opt.step();torch.mps.synchronize()
                clear(sources);steps+=1
                if steps%4==0:
                    print("G44_PROGRESS",json.dumps({"updates":steps,"unique_train_seen":len(visited),"elapsed":time.perf_counter()-start,"loss":total/tokens}),flush=True)
                    (out/f"step_{steps}.safetensors").write_bytes(serialize())
                assert all(p.grad is None and p._version==versions[id(p)] for p in frozen)
            assert len(visited)==512
            packet=serialize();(out/f"epoch_{ep}.safetensors").write_bytes(packet)
            record={"epoch":ep,"loss":total/tokens,"updates":steps,"nonzero_gradient_groups":sorted(seen),"gpu_elapsed_seconds":time.perf_counter()-start,"unique_train_seen":len(visited),"budgets":balance.budgets().detach().cpu().tolist() if arm=="balanced" else None}
            curve.append(record);print("G44_EPOCH",arm,json.dumps(record),flush=True)
        assert {"head","slot","core","GT","l6"}<=seen,seen
        changes={k:float((p.detach()-init[k]).norm()) for k,p in selected.items()}
        assert all(any(changes[k]>0 for k in selected if groups[k]==g) for g in set(groups.values()))
        final=(out/f"epoch_{EPOCHS}.safetensors").read_bytes();fields=load(final)
        opt.zero_grad(set_to_none=True);del opt,init
        for s in sources.values():s.clear()
        for m in modules:m.to("cpu")
        for s in sources.values():s.clear();s.to("cpu")
        selected={"head."+k:p for k,p in child.language_head.named_parameters()}|{"slot."+k:p for k,p in slot.named_parameters()}|{"l6."+k:p for k,p in l6.named_parameters()}|{"core."+k:sources[k].centroids for k in keys}|{"GT.gates":full.gates}
        with torch.no_grad():
            for k,p in selected.items():p.copy_(fields[k].float())
        for prefix,mod in (("head.",child.language_head),("slot.",slot),("l6.",l6)):
            mod.load_state_dict({k[len(prefix):]:v for k,v in fields.items() if k.startswith(prefix)},strict=True)
        clear(sources);after,items=evaluate(st,tok,train,slot)
        replay,replay_items=evaluate(st,tok,train[:32],slot)
        old_after,old_results=evaluate(st,tok,dev,slot)
        fresh_result,extra_items=evaluate(st,tok,fresh,slot)
        (out/"REPLAY_AND_OLD_DEV.json").write_text(json.dumps(replay_items+old_results,ensure_ascii=False))
        (out/"FRESH_DEV.json").write_text(json.dumps(extra_items,ensure_ascii=False))
    (out/"FINAL_DEV.json").write_text(json.dumps(items,ensure_ascii=False))
    (out/"g44_joint.safetensors").write_bytes(final)
    size=sum(map(len,blobs))+sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())+len(final)
    bpp=8*size/G05;assert bpp<=.7
    report={"arm":arm,"parent_checkpoint_sha":PARENT[arm],"checkpoint_sha":digest(final),"baseline_replay":before,"baseline_old_dev":old_before,"baseline_fresh":fresh_before,"train":after,"replay":replay,"old_dev":old_after,"fresh":fresh_result,"learned_budgets":balance.budgets().detach().cpu().tolist() if arm=="balanced" else None,"curve":curve,"train_letters":2,"train_prompts":512,"old_dev_prompts":32,"fresh_prompts":64,"known_payload_bpp":bpp,"nonzero_gradient_groups":sorted(seen),"replay_gate":all(d["exact_EOS"]==32 for d in replay.values()),"train_gate":all(d["exact_EOS"]==512 for d in after.values()),"fresh95_gate":all(d["exact_EOS"]>=61 for d in fresh_result.values()),"candidate_accepted":False,"full_mps_training":True,"CPU_fallback_disabled":True,"gpu_weight_changes":changes,"CPU_GPU_initial_logit_error":err,"limitations":["One full MPS pass through 512 distinct TRAIN prompts, plus 64 extra replay exposures. Existing head, slot, L6 adapter, L6/L7 compact codebooks and GT gates train; NRA3 and learned balance gains stay frozen. No new parameters.","480 new TRAIN prompts are factorial combinations of 40 authored instruction frames, six delimiters and two letters, not 480 independent semantic tasks. Fresh 64 prompts are exploratory, fixed before training and never used for weight selection.","No 616 skills or fixed-work CPU speed certification. G43 remains retained parent if any gate fails."]}
    (out/"G44_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2));print("G44_REPORT",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--arm",choices=("balanced",),required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.out,a.arm)
