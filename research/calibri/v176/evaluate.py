"""Actual saved native candidate, original model, and native baseline."""
import statistics
from v175.evaluate import generate, summarize
from v174.answers import annotate
from .protocol import ARMS


def answers(models, tokenizer, cases, progress):
    records = {a: [] for a in ARMS}
    for i, case in enumerate(cases):
        hashes = []
        for arm in ARMS[i % 3:] + ARMS[:i % 3]:
            output = generate(models[arm], tokenizer, case['prompt'])
            records[arm].append(annotate(case, output))
            hashes.append(output['prompt_sha'])
        if len(set(hashes)) != 1:
            raise AssertionError('Different questions between models')
        if (i+1) % 8 == 0:
            progress(records, i+1)
    return records


def speed(models, tokenizer, cases):
    chosen = [c for c in cases if c['question_id'] in (0, 16, 64, 80)]
    for arm in ('CB59', 'CB176'):
        generate(models[arm], tokenizer, chosen[0]['prompt'], cap=8, stop=False)
    rows = []
    for repeat in range(2):
        for i, case in enumerate(chosen):
            order = ('CB59', 'CB176') if (i+repeat) % 2 == 0 else ('CB176', 'CB59')
            times = {}
            for arm in order:
                result = generate(models[arm], tokenizer, case['prompt'], cap=32, stop=False)
                if len(result['token_ids']) != 32:
                    raise AssertionError('Unequal benchmark work')
                times[arm] = result['wall_seconds']
            rows.append({'repeat': repeat, 'question_id': case['question_id'], 'seconds': times,
                         'candidate_speed_ratio': times['CB59']/times['CB176']})
    return {'pairs': rows, 'fixed_tokens': 32, 'includes_prefill': True,
            'median_speed_ratio': statistics.median(r['candidate_speed_ratio'] for r in rows)}

