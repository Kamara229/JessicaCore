"""Real native answers and separate matched fixed-work CPU timings."""
import hashlib
import time
import statistics
import torch
from v174.answers import annotate, summary
from .protocol import ARMS, MAX_NEW_TOKENS, SPEED_CASE_IDS

@torch.inference_mode()
def generate(model, tok, question, cap=MAX_NEW_TOKENS, stop=True):
    prompt = tok.apply_chat_template([{'role':'user','content':question}], tokenize=False, add_generation_prompt=True)
    if question not in prompt or tok.eos_token_id is None:
        raise ValueError('Invalid question/template/EOS')
    x = torch.tensor([tok.encode(prompt, add_special_tokens=False)], dtype=torch.long)
    if not 1 < x.shape[1] < 256:
        raise ValueError('Unexpected prompt size')
    cache, emitted = None, []
    start = time.perf_counter()
    for i in range(cap):
        result = model(input_ids=x, past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        logits = result.logits[:, -1, :]
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite native logits')
        x = logits.argmax(dim=-1, keepdim=True)
        emitted.append(int(x.item()))
        if i == 0:
            prefill = time.perf_counter() - start
        if stop and emitted[-1] == tok.eos_token_id:
            break
    elapsed = time.perf_counter() - start
    ended = emitted[-1] == tok.eos_token_id
    return {'verbatim_text':tok.decode(emitted, skip_special_tokens=True),
            'token_ids':emitted, 'eos':ended,
            'cap_hit_without_eos':not ended and len(emitted)==cap,
            'prompt_sha':hashlib.sha256(prompt.encode()).hexdigest(),
            'wall_seconds':elapsed, 'prefill_seconds':prefill,
            'distinct_token_fraction':len(set(emitted))/len(emitted)}


def answers(models, tok, cases, save_progress):
    records = {a:[] for a in ARMS}
    for i, case in enumerate(cases):
        hashes = []
        for arm in ARMS[i%3:] + ARMS[:i%3]:
            output = generate(models[arm], tok, case['prompt'])
            records[arm].append(annotate(case, output))
            hashes.append(output['prompt_sha'])
        if len(set(hashes)) != 1:
            raise AssertionError('Different prompts across models')
        if (i+1)%16 == 0:
            save_progress(records, i+1)
    return records


def matched_speed(models, tok, cases):
    # Same prompt and exactly 32 token steps even if EOS appears. This is a
    # throughput diagnostic, kept separate from quality generation to EOS.
    chosen = [c for c in cases if c['question_id'] in SPEED_CASE_IDS]
    for arm in ('CB59','CB175'):
        generate(models[arm], tok, chosen[0]['prompt'], cap=8, stop=False)
    rows=[]
    for repeat in range(2):
        for i, case in enumerate(chosen):
            order=('CB59','CB175') if (i+repeat)%2==0 else ('CB175','CB59')
            times={}
            for arm in order:
                result=generate(models[arm],tok,case['prompt'],cap=32,stop=False)
                if len(result['token_ids']) != 32:
                    raise AssertionError('Unequal timing work')
                times[arm]=result['wall_seconds']
            rows.append({'repeat':repeat,'question_id':case['question_id'],
                         'seconds':times,'candidate_speed_ratio':times['CB59']/times['CB175']})
    return {'fixed_tokens':32,'pairs':rows,
            'median_speed_ratio':statistics.median(r['candidate_speed_ratio'] for r in rows),
            'includes_prefill':True,'not_answer_quality':True}


def summarize(records):
    out={}
    for arm, rows in records.items():
        out[arm]={'all':summary(rows),
                   'by_domain':{d:summary([r for r in rows if r['domain']==d])
                                for d in sorted({r['domain'] for r in rows})}}
    return out
