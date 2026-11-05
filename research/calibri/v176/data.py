"""Freeze public TRAIN text and distinct supervised examples before fitting."""
import hashlib
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from quality import DATASET, load_token_ids
from paid_lowrank_v76 import AG_DATASET
from .protocol import MAX_TRAIN_LENGTH


def instruction_pairs():
    rows = []
    # New values, names, objects and formulations; no historical QA labels.
    for i in range(8):
        a, b = 210 + i * 7, 101 + i * 3
        rows.append((f'What do you get when you add {a} and {b}?', str(a+b)))
        rows.append((f'Subtract {b} from {a}.', str(a-b)))
    for i in range(8):
        a, b = f'PersonA{i}', f'PersonB{i}'
        rows.append((f'{a} holds a key. {b} holds a coin. Who holds the key?', a))
        answer = 'Yes' if i % 2 == 0 else 'No'
        target = a if i % 2 == 0 else b
        rows.append((f'{a} holds a key. {b} holds a coin. Does {target} hold the key?', answer))
    assert len(rows) == 32 and len(set(q for q, _ in rows)) == 32
    return rows


def prepare(tokenizer):
    samples, provenance = [], {}
    # Exactly 2,048 public TRAIN tokens, 32 windows, plus 32 short tasks.
    for name, dataset in [('wiki', DATASET), ('news', AG_DATASET)]:
        revision = HfApi().dataset_info(dataset).sha
        if name == 'wiki':
            ids = load_token_ids(tokenizer, 'train', revision, 31024)[30000:31024]
            origin = {'token_span': [30000, 31024]}
        else:
            records = load_dataset(dataset, split='train[18000:18100]', revision=revision)
            raw = '\n\n'.join(str(r['text']) for r in records)
            ids = tokenizer(raw, add_special_tokens=False, return_tensors='pt').input_ids[0, :1024]
            origin = {'record_range': [18000, 18100], 'token_span': [0, 1024]}
        if len(ids) != 1024:
            raise ValueError('Incomplete public TRAIN text')
        provenance[name] = {'dataset': dataset, 'split': 'train', 'revision': revision,
                            **origin, 'token_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
        for j in range(0, len(ids), 64):
            samples.append({'ids': ids[j:j+64].clone(), 'kind': name, 'answer_start': None})
    for question, answer in instruction_pairs():
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                    tokenize=False, add_generation_prompt=True)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suffix = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = torch.tensor(prefix + suffix, dtype=torch.long)
        if len(ids) > MAX_TRAIN_LENGTH or tokenizer.eos_token_id is None:
            raise ValueError('Unexpected supervised chat sequence')
        samples.append({'ids': ids, 'kind': 'supervised_chat', 'answer_start': len(prefix),
                        'question': question, 'answer': answer})
    assert len(samples) == 64
    provenance['supervised_chat'] = [{'question': q, 'answer': a} for q, a in instruction_pairs()]
    provenance['all_input_sha256'] = hashlib.sha256(b''.join(s['ids'].numpy().tobytes() for s in samples)).hexdigest()
    provenance['sample_count'] = len(samples)
    provenance['tokens_per_epoch'] = sum(len(s['ids']) for s in samples)
    return samples, provenance


@torch.no_grad()
def capture(teacher, samples):
    references = []
    for sample in samples:
        ids = sample['ids'][None]
        result = teacher(input_ids=ids, use_cache=False, output_hidden_states=True)
        target = result.logits.argmax(-1).detach().clone()
        mask = torch.ones_like(target, dtype=torch.bool)
        if sample['answer_start'] is not None:
            # At position prefix_len-1, predict the FIRST answer token.
            first = sample['answer_start'] - 1
            mask.zero_(); mask[:, first:-1] = True
            target[:, first:-1] = ids[:, first+1:]
        references.append({'hidden': tuple(h.detach().clone() for h in result.hidden_states),
                           'targets': target, 'mask': mask})
    return references

