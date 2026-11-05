"""Joint compressed-model training, byte seal, native QA and matched speed."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v173.run import matrix_roundtrip
from . import protocol as p
from .data import prepare, capture
from .operators import build, clear
from .codec import export
from .train import fit
from .evaluate import answers, speed, summarize


def run(source, out):
    torch.set_num_threads(2)
    torch.manual_seed(p.SEED)
    out.mkdir(parents=True, exist_ok=True)
    archive = extract_original(source)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    samples, provenance = prepare(tokenizer)
    cases = freeze_cases(all_cases())
    if {s.get('question') for s in samples} & {c['question'] for c in cases}:
        raise AssertionError('Training/evaluation question collision')
    frozen = {'experiment': 'v176_joint_paid_codebook_scale_norm_training',
              'model': MODEL, 'revision': MODEL_SHA, 'source_sha': archive.hash(),
              'source_bytes': p.SOURCE_BYTES, 'seed': p.SEED, 'steps': p.STEPS,
              'lr': p.LR, 'norm_lr': p.NORM_LR, 'hidden_weight': p.HIDDEN_WEIGHT,
              'margin_weight': p.MARGIN_WEIGHT, 'margin': p.MARGIN,
              'training': provenance, 'evaluation': cases, 'arms': p.ARMS,
              'fixed_final_checkpoint': True, 'QA_never_used_for_training_or_selection': True,
              'historical_development_suite_not_unseen_TEST': True,
              'no_KL_NLL_PPL_gate': True, 'no_production_changes': True,
              'fixed_indices_only_existing_continuous_fields_updated': True}
    save(out/'protocol_before_fit.json', frozen)
    teacher = original_teacher()
    references = capture(teacher, samples)
    print(json.dumps({'stage': 'teacher_TRAIN_captured', 'samples': len(samples)}), flush=True)
    student, sources = build(archive)
    # Validate the trainable surrogate against the original native checkpoint
    # before any updates, on a TRAIN window, not a quality-test question.
    with TemporaryDirectory(dir=out) as tmp:
        folder = Path(tmp)
        original = folder/'source.cb59'; original.write_bytes(archive.blob)
        native, _, _ = make_model('baseline', SimpleNamespace(original=original, folder=folder))
        with torch.no_grad():
            ids = samples[0]['ids'][:8][None]
            expected = native(input_ids=ids, use_cache=False).logits
            actual = student(input_ids=ids, use_cache=False).logits
            err = float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))
            if err > 1e-4 or not torch.isfinite(actual).all():
                raise AssertionError('Initial TRAIN surrogate/native mismatch')
        clear(sources)
        del native, actual, expected
    save(out/'initial_surrogate_audit.json', {'relative_logit_error': err})
    def checkpoint(step, curve):
        blob = export(archive, student, sources)
        (out/f'TRAIN_checkpoint_{step:03d}.cb59').write_bytes(blob)
        save(out/'training_progress.json', {'steps_completed': step, 'curve': curve,
                                           'checkpoint_sha': hashlib.sha256(blob).hexdigest()})
    training = fit(student, sources, samples, references, checkpoint)
    blob = export(archive, student, sources)
    candidate = out/'CALIBRI_v176_joint_same_bytes.cb59'; candidate.write_bytes(blob)
    saved = CB59(candidate.read_bytes())
    if len(blob) != p.SOURCE_BYTES:
        raise AssertionError('Physical source budget exceeded')
    # Save one held TRAIN window's logits from the final surrogate so the
    # persisted native model can be compared after all training state is gone.
    with torch.no_grad():
        clear(sources)
        probe_ids = samples[0]['ids'][:8][None]
        trained_logits = student(input_ids=probe_ids, use_cache=False).logits.detach().clone()
        clear(sources)
    changed = {'centroids': 0, 'row_scales': 0, 'raw_vectors': 0}
    for name, d in archive.meta['tensor_descriptors'].items():
        if d['dtype'] == 'CB59_K32V8':
            for part in ('centroids', 'row_scales'):
                changed[part] += int(not np.array_equal(getattr(archive.handles[name], part), getattr(saved.handles[name], part)))
        else:
            changed['raw_vectors'] += int(not np.array_equal(archive.raw[name], saved.raw[name]))
    sealed = {'sha256': saved.hash(), 'bytes': len(blob), 'bpp': 8*len(blob)/p.TOTAL_UNIQUE,
              'changed_fields': changed, 'training': training, 'sealed_before_QA': True,
              'no_sidecar_or_teacher_needed_for_inference': True}
    save(out/'candidate_sealed_before_QA.json', sealed)
    del student, sources, references, samples
    gc.collect()
    with TemporaryDirectory(dir=out) as tmp:
        root = Path(tmp)
        original = root/'CB59.cb59'; original.write_bytes(archive.blob)
        models = {'FP32': teacher}
        for arm, path in [('CB59', original), ('CB176', candidate)]:
            folder = root/arm; folder.mkdir()
            model, meta, _ = make_model('baseline', SimpleNamespace(original=path, folder=folder))
            if not meta['zero_dense_pretrained_2D_hot_weights'] or any(p.ndim >= 2 for p in model.parameters()):
                raise AssertionError('Dense trained matrix in compressed inference')
            if model.model.embed_tokens.handle is not model.lm_head.handle:
                raise AssertionError('Broken saved tied source')
            models[arm] = model
        with torch.no_grad():
            actual = models['CB176'](input_ids=probe_ids, use_cache=False).logits
            final_error = float(torch.linalg.vector_norm(actual-trained_logits)/torch.linalg.vector_norm(trained_logits))
            if final_error > 1e-4 or not torch.isfinite(actual).all():
                raise AssertionError('Trained surrogate/export/native mismatch')
        native_audit = matrix_roundtrip(saved, models['CB176'].lm_head)
        native_audit['trained_surrogate_relative_logit_error'] = final_error
        save(out/'native_roundtrip.json', native_audit)
        timings = speed(models, tokenizer, cases)
        save(out/'matched_native_speed.json', timings)
        print(json.dumps({'stage': 'native_speed', 'ratio': timings['median_speed_ratio']}), flush=True)
        def progress(records, done):
            save(out/'answers_partial.json', {'completed_cases': done, 'answers': records})
            print(json.dumps({'stage': 'real_answers', 'questions_per_arm': done}), flush=True)
        records = answers(models, tokenizer, cases, progress)
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != sealed['sha256']:
        raise AssertionError('Candidate changed during QA')
    result = {'protocol': frozen, 'candidate': sealed, 'native_audit': native_audit,
              'native_speed': timings, 'summary': summarize(records), 'answers': records,
              'complete': True, 'generations': sum(len(x) for x in records.values()),
              'ungraded_answers_require_semantic_review': True,
              'training_loss_reduction_is_not_a_quality_pass': True}
    save(out/'v176_results.json', result)
    print(json.dumps({'stage': 'complete', 'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path); parser.add_argument('out', type=Path)
    args = parser.parse_args(); run(args.source, args.out)
