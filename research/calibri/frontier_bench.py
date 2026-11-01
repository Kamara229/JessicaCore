"""Reproducible synthetic mechanics check. NOT a pretrained-model benchmark."""
import argparse
import json
import pathlib
from dataclasses import asdict
import numpy as np
from precision_frontier import Candidate, allocate
from residual_page import pack_page, unpack_page


def run(seed=20261101):
    rng = np.random.default_rng(seed)
    # Disjoint output rows make squared linear output errors additive.
    weights = rng.normal(size=(96, 128)) * np.repeat([.15, .3, .6, 1., 2., 4.], 16)[:, None]
    def inputs():
        common = rng.normal(size=(128, 128))
        rare = rng.normal(size=(128, 128))
        rare[:, -16:] *= 4
        return (common, rare)
    calibration, heldout = inputs(), inputs()
    rms = np.sqrt(np.mean(np.concatenate(calibration) ** 2, axis=0))
    cal_energy = [np.sum((x @ weights.T) ** 2) for x in calibration]
    eval_energy = [np.sum((x @ weights.T) ** 2) for x in heldout]
    configurations = [('sparse', .885, 0.), ('residual1', .885, .01),
                      ('residual5', .885, .05), ('uniform75', .75, 0.),
                      ('uniform50', .50, 0.)]
    pages, restored, byte_counts = [], [], []
    for start in range(0, len(weights), 16):
        w = weights[start:start+16]
        variants, sizes = {}, {}
        for name, zeros, corrections in configurations:
            blob = pack_page(w, rms, zeros, corrections)
            variants[name] = unpack_page(blob)
            sizes[name] = len(blob)
        # Dense fallback uses 1-byte mode + 2 uint16 dimensions + FP16 payload.
        variants['fp16'] = w.astype('<f2').astype(np.float64)
        sizes['fp16'] = 5 + 2 * w.size
        candidates = []
        for name, value in variants.items():
            errors = tuple(float(np.sum((x @ (w-value).T) ** 2) / energy)
                           for x, energy in zip(calibration, cal_energy))
            candidates.append(Candidate(name, sizes[name], errors))
        pages.append(candidates)
        restored.append(variants)
        byte_counts.append(sizes)
    baseline = [next(c for c in p if c.name == 'uniform75') for p in pages]
    limits = tuple(sum(c.errors[i] for c in baseline) + 1e-12 for i in range(2))
    chosen = allocate(pages, limits)
    def heldout_error(names):
        result = np.vstack([variants[n] for variants, n in zip(restored, names)])
        return [float(np.sum((x @ (weights-result).T) ** 2) / energy)
                for x, energy in zip(heldout, eval_energy)]
    baseline_bytes = sum(c.byte_count for c in baseline)
    # Same enclosing index cost for all allocations: header + offset/size per page.
    container_bytes = 16 + 12 * len(pages)
    return {'data': 'SYNTHETIC Gaussian weights and two activation strata; no pretrained model',
            'seed': seed, 'weights': int(weights.size), 'pages': len(pages),
            'baseline': {'page_bytes': baseline_bytes, 'total_bytes': baseline_bytes + container_bytes,
                         'calibration_error_limits': limits,
                         'heldout_relative_squared_output_error': heldout_error(['uniform75']*len(pages))},
            'allocated': {**asdict(chosen), 'total_bytes': chosen.byte_count + container_bytes,
                          'heldout_relative_squared_output_error': heldout_error(chosen.choices)},
            'saving_percent': 100*(1-(chosen.byte_count+container_bytes)/(baseline_bytes+container_bytes)),
            'limits': ['synthetic only', 'linear-output proxy, not perplexity',
                       'no latency/RAM/Android measurement', 'dense fallback and outer index accounted, not serialized',
                       'heldout was evaluated only after selection; no heldout acceptance bound claimed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/calibri_v12_synthetic.json')
    args = parser.parse_args()
    result = run()
    p = pathlib.Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
