"""G13: read-only comparison of actual G05 CALIBRI and open SmolLM2-135M-Instruct.
No model used as teacher. Authored bilingual diagnostic, fixed prompts,
equal output-token cap, native templates, CPU timings and raw answers.
Explicitly distinguish exact short-answer score from general chat quality.
"""
import argparse
import json
import re
import statistics
import tempfile
import time
import unicodedata
from pathlib import Path
import torch
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import open_bundle,digest
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g11_half_cache_cpu import install_ffn
from research.calibri2.fusion_growth.g12_scaled_cpu_reference import cache_attention
from research.calibri2.fusion_growth.f21_native_cpu_integration import (
    build_v2,install_packed,last_forward,keep_keyword,prompt_features,strict_chat_source_ids)

REPO="HuggingFaceTB/SmolLM2-135M-Instruct"
REV="5a33ba103645800d7b3790c4448546c1b73efc71"
CAP=16
MODES=("CALIBRI_PACKED","CALIBRI_ALL16","SMOL_FP32_KV","SMOL_BF16_KV")

def suite():
    cases={
      "arithmetic":[
        ("23 + 14 = ?", "37","What is 23 + 14?","37"),
        ("Вычисли 48 − 19.","29","Calculate 48 minus 19.","29"),
        ("Сколько будет 7 умножить на 6?","42","What is 7 times 6?","42"),
        ("Раздели 84 на 7.","12","Divide 84 by 7.","12")],
      "logic":[
        ("Все лисы — животные. Руди — лиса. Руди является животным?","да","All foxes are animals. Rudy is a fox. Is Rudy an animal?","yes"),
        ("Ни одна птица не является рыбой. Сова — птица. Сова является рыбой?","нет","No bird is a fish. An owl is a bird. Is an owl a fish?","no"),
        ("Павел старше Анны, Анна старше Иры. Кто старше всех?","Павел","Paul is older than Anna, Anna is older than Ira. Who is oldest?","Paul"),
        ("Если 15 больше 9, верно ли что 9 больше 15?","нет","If 15 is greater than 9, is 9 greater than 15?","no")],
      "context":[
        ("В коробке лежит монета, а в шкафу чашка. Где монета?","в коробке","A coin is in the box and a cup is in the cupboard. Where is the coin?","in the box"),
        ("Роза отдала книгу Вите. У кого теперь книга?","Витя","Rose gave the book to Victor. Who now has the book?","Victor"),
        ("После среды наступил четверг. Какой день был первым из двух?","среда","Thursday came after Wednesday. Which of the two came first?","Wednesday"),
        ("У Нади три яблока. Она отдала одно. Сколько осталось?","2","Nadia has three apples and gives one away. How many remain?","2")],
      "facts":[
        ("Назови столицу Италии.","Рим","What is the capital of Italy?","Rome"),
        ("Сколько дней в неделе?","7","How many days are in a week?","7"),
        ("Какая планета известна как Красная планета?","Марс","Which planet is known as the Red Planet?","Mars"),
        ("Как называется замёрзшая вода?","лёд","What is frozen water called?","ice")],
      "instructions":[
        ("Повтори точно: белый парус.","белый парус","Repeat exactly: white sail.","white sail"),
        ("Напиши второе слово: ветер море лес.","море","Write the second word: wind sea forest.","sea"),
        ("Напиши цифры 4 и 8 подряд без пробела.","48","Write digits 4 and 8 together without a space.","48"),
        ("Из слов груша, велосипед, слива выбери транспорт.","велосипед","From pear, bicycle, plum choose the vehicle.","bicycle")],
      "sequence":[
        ("Продолжи ряд: 2, 4, 6, 8, ...","10","Continue the sequence: 2, 4, 6, 8, ...","10"),
        ("Продолжи ряд: 5, 10, 15, 20, ...","25","Continue the sequence: 5, 10, 15, 20, ...","25"),
        ("Какое число следует за 99?","100","What number comes after 99?","100"),
        ("Какой день следует за пятницей?","суббота","What day follows Friday?","Saturday")]}
    rows=[]
    for domain,items in cases.items():
        for i,(rq,ra,eq,ea) in enumerate(items):
            for lang,q,a,suffix in (("ru",rq,ra," Ответь кратко, только ответ."),("en",eq,ea," Give only the short answer.")):
                rows.append({"id":f"{domain}/{i}/{lang}","domain":domain,"language":lang,"q":q+suffix,"gold":a})
    assert len(rows)==48 and len({r["q"] for r in rows})==48
    return rows

def norm(s):
    s=unicodedata.normalize("NFKC",s).casefold().replace("ё","е").strip()
    return re.sub(r"\s+"," ",s).strip(" .!?\n\t\"'«»")

def calibri_generate(st,key,tok,q):
    p=strict_chat_source_ids(tok,q);f=prompt_features(tok,q)
    ids=[];t=time.perf_counter()
    for _ in range(CAP):
        z=last_forward(st,p+ids,f,key)
        ids.append(int(z.argmax()))
        if ids[-1]==2:break
    return tok.decode(ids,skip_special_tokens=True).strip(),ids,time.perf_counter()-t,ids[-1]==2,len(p)

def smol_generate(model,tok,q):
    p=tok.apply_chat_template([{"role":"user","content":q}],tokenize=True,add_generation_prompt=True,return_tensors="pt")
    t=time.perf_counter()
    with torch.inference_mode():
        out=model.generate(p,attention_mask=torch.ones_like(p),max_new_tokens=CAP,
                           do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    dt=time.perf_counter()-t
    ids=out[0,p.shape[-1]:].tolist()
    eos=model.generation_config.eos_token_id
    stop=set(eos if isinstance(eos,list) else [eos])
    return tok.decode(ids,skip_special_tokens=True).strip(),ids,dt,ids[-1] in stop,p.shape[-1]

def run(g05,f08,out):
    torch.manual_seed(20270105);torch.set_num_threads(2)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    ar,_,_=open_bundle(blobs[0])
    ctok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    # Only official public weights; no remote code, no instruction output
    # ever used to train or tune CALIBRI.
    root=Path(snapshot_download(REPO,revision=REV,allow_patterns=[
        "config.json","generation_config.json","model.safetensors","tokenizer.json",
        "tokenizer_config.json","special_tokens_map.json","vocab.json","merges.txt"]))
    stok=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    assert ctok.vocab_size==stok.vocab_size==49152
    out.mkdir(parents=True,exist_ok=False)
    rows=suite();(out/"G13_SUITE.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    states={};keys={};models={};inventory={};records=[]
    with tempfile.TemporaryDirectory() as td:
        lib,kernel=build_v2(td)
        for mode in MODES[:2]:
            st,second,added,fourth,reg,old,fifth,sixth,reg2=restore(blobs[:6],blobs[6],blobs[7])
            install_packed(st,ar,kernel);keys[mode]=keep_keyword(st[0]);states[mode]=st
            inv={"logical_core_entries":G05,"weight_payload_bytes":sum(map(len,blobs)),
                 "tokenizer_bytes":sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file()),
                 "additional_cache_bytes":0}
            if mode=="CALIBRI_ALL16":
                v,_=install_ffn(st,(st[1],second,added,fourth,fifth,sixth),"FUSED_FP16")
                inv["additional_cache_bytes"]=v["FFN_cache_bytes"]+cache_attention(st)["attention_cache_bytes"]
            inventory[mode]=inv
            del second,added,fourth,reg,old,fifth,sixth,reg2
        for mode,dtype in zip(MODES[2:],(torch.float32,torch.bfloat16)):
            m=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,trust_remote_code=False,
                                                  torch_dtype=dtype,attn_implementation="eager").eval()
            models[mode]=m
            inventory[mode]={"independent_dense_parameters":sum(p.numel() for p in m.parameters()),
                             "weight_file_bytes":(root/"model.safetensors").stat().st_size,
                             "package_bytes":sum(p.stat().st_size for p in root.iterdir() if p.is_file()),
                             "resident_parameter_tensor_bytes":sum(p.numel()*p.element_size() for p in m.parameters()),
                             "inference_dtype":str(dtype),"KV_cache":True}
        def gen(m,q):
            return calibri_generate(states[m],keys[m],ctok,q) if m.startswith("CALIBRI") else smol_generate(models[m],stok,q)
        for m in MODES:gen(m,"Write only the number 6.")
        for i,row in enumerate(rows):
            order=MODES[i%len(MODES):]+MODES[:i%len(MODES)]
            ans={}
            for m in order:
                text,ids,seconds,eos,prompt_tokens=gen(m,row["q"])
                ans[m]={"text":text,"ids":ids,"seconds":seconds,"eos":eos,"prompt_tokens":prompt_tokens,
                        "strict_exact":text==row["gold"],"normalized_exact":norm(text)==norm(row["gold"])}
            records.append({**row,"answers":ans})
            if (i+1)%8==0:print("G13_PROGRESS",i+1,flush=True)
        result={}
        for m in MODES:
            rr=[r["answers"][m] for r in records]
            result[m]={**inventory[m],"n":len(rr),"strict_exact":sum(x["strict_exact"] for x in rr),
                       "normalized_exact":sum(x["normalized_exact"] for x in rr),
                       "completed_EOS":sum(x["eos"] for x in rr),
                       "generated_tokens":sum(len(x["ids"]) for x in rr),
                       "tokens_per_second_including_prefill":sum(len(x["ids"]) for x in rr)/sum(x["seconds"] for x in rr),
                       "median_seconds_per_answer":statistics.median(x["seconds"] for x in rr),
                       "by_domain":{d:{"n":sum(r["domain"]==d for r in records),
                                        "exact":sum(r["domain"]==d and r["answers"][m]["normalized_exact"] for r in records)}
                                    for d in sorted({r["domain"] for r in rows})},
                       "by_language":{l:{"n":sum(r["language"]==l for r in records),
                                          "exact":sum(r["language"]==l and r["answers"][m]["normalized_exact"] for r in records)}
                                      for l in ("ru","en")}}
            print("G13_ARM",m,json.dumps(result[m],sort_keys=True),flush=True)
    report={"stage":"G13_OPEN_135M_DIRECT_BENCHMARK","open_model":REPO,"open_revision":REV,
            "max_new_tokens":CAP,"threads":2,"suite_sha256":digest(json.dumps(rows,ensure_ascii=False).encode()),
            "arms":result,"records":records,
            "CALIBRI_optimization_sequence_parity":sum(r["answers"]["CALIBRI_PACKED"]["ids"]==r["answers"]["CALIBRI_ALL16"]["ids"] for r in records),
            "candidate_accepted":False,"training_updates":0,"teacher_calls":0,
            "limitations":["SmolLM2 English-first; report languages separately. Small authored diagnostic, related skill families from prior training, not a standard blind benchmark. Exact scoring can reject semantically correct verbose or inflected answers; raw outputs retained.",
                           "CALIBRI logical matrix entries are not independent dense parameters. Its stored payload excludes executable runtime; caches and working memory separate.",
                           "Native chat templates, equal output token cap; prompt token counts and output lengths may differ. SmolLM2 KV cache vs CALIBRI full-prefix path; timing describes these actual research runtimes, not architecture-only speed.",
                           "No public-model outputs used as targets or training data; no protected historical 616 replay, no deployment."]}
    (out/"G13_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G13_ASSESS",json.dumps({"stage":report["stage"],"CALIBRI_optimization_sequence_parity":report["CALIBRI_optimization_sequence_parity"],"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
