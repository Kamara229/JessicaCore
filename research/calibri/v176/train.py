"""Joint gradients through all 30 compressed blocks; fixed final iterate."""
import json
import random
import time
import torch
from .operators import clear
from .objective import objective
from .protocol import SEED, STEPS, LR, NORM_LR, CHECKPOINT_EVERY


def fit(model, sources, samples, references, checkpoint):
    source_params = [p for source in sources.values() for p in source.parameters()]
    source_ids = {id(p) for p in source_params}
    norms = [p for p in model.parameters() if id(p) not in source_ids]
    if len(source_ids) != len(source_params):
        raise AssertionError('Duplicate optimizer parameter')
    optimizer = torch.optim.Adam([{'params': source_params, 'lr': LR},
                                   {'params': norms, 'lr': NORM_LR}])
    rng = random.Random(SEED)
    order, curve = [], []
    started = time.perf_counter()
    for step in range(STEPS):
        if step % len(samples) == 0:
            order = list(range(len(samples))); rng.shuffle(order)
        index = order[step % len(samples)]
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids=samples[index]['ids'][None], use_cache=False, output_hidden_states=True)
        loss, hidden, margin = objective(result, references[index])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite joint TRAIN objective')
        loss.backward()
        if step == 0:
            missing = [n for n, s in sources.items() if s.centroids.grad is None or s.log_scales.grad is None]
            if missing or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise AssertionError('Joint source gradient absent/nonfinite: ' + str(missing))
        gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for s in sources.values():
                s.log_scales.clamp_(-1.0, 1.0)
                s.centroids.clamp_(-16, 16)
        item = {'step': step+1, 'sample': index, 'kind': samples[index]['kind'],
                'TRAIN_objective': float(loss.detach()), 'TRAIN_hidden_error': float(hidden.detach()),
                'TRAIN_hard_token_margin': float(margin.detach()), 'gradient_norm': float(gradnorm),
                'elapsed_seconds': time.perf_counter()-started}
        curve.append(item)
        del result, loss, hidden, margin
        clear(sources)
        if (step+1) % 8 == 0:
            print(json.dumps({'stage': 'joint_TRAIN', **item}), flush=True)
        if (step+1) % CHECKPOINT_EVERY == 0:
            checkpoint(step+1, curve)
    return {'steps': STEPS, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
            'matrix_count': len(sources), 'curve': curve,
            'selection': 'fixed final step; no QA/VALID-selected checkpoint',
            'no_KL_NLL_PPL_objective': True}
