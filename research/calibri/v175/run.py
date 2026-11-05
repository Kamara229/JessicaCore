"""Full source rewrite, native verification and 288 actual answers."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v173.run import matrix_roundtrip
from direct_codebook_v59 import CB59
from .protocol import SOURCE_BYTES, SOURCE_SHA, TOTAL_UNIQUE, ARMS, DAMPING, TIED_EMBEDDING_PENALTY
from .calibration import prepare_text, collect
from .codec import recode
from .evaluate import answers, matched_speed, summarize


def run(source, out):
    torch.set_num_threads(2)
    torch.manual_seed(175)
    out.mkdir(parents=True,exist_ok=True)
    cases=freeze_cases(all_cases())
    windows, provenance=prepare_text()
    frozen={'stage':'v175_full_covariance_index_feedback_same_bytes',
            'model':MODEL,'revision':MODEL_SHA,'source_sha':SOURCE_SHA,
            'source_bytes':SOURCE_BYTES,'bpp':8*SOURCE_BYTES/TOTAL_UNIQUE,
            'fit_text':provenance,'fit_windows':len(windows),
            'fit_token_count':sum(len(x) for x in windows),
            'cases':cases,'arms':ARMS,'damping':DAMPING,
            'tied_embedding_penalty':TIED_EMBEDDING_PENALTY,
            'max_answer_tokens':64,'no_KL_NLL_PPL_gate':True,
            'no_private_v89_TEST':True,'historical_development_questions':True,
            'same_centroids_scales_and_native_decoder':True,
            'no_production_changes':True}
    save(out/'protocol_before_fit.json',frozen)
    archive=extract_original(source)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=original_teacher()
    moments, counts=collect(teacher,archive,windows)
    print(json.dumps({'stage':'covariance_captured','shared_inputs':len(moments)}),flush=True)
    blob, fit=recode(archive,teacher,moments)
    del moments
    gc.collect()
    candidate=out/'CALIBRI_v175_same_bytes.cb59'
    candidate.write_bytes(blob)
    verified=CB59(candidate.read_bytes())
    if len(blob)!=SOURCE_BYTES or verified.hash()!=hashlib.sha256(blob).hexdigest():
        raise AssertionError('Persisted source mismatch')
    sealed={'candidate_sha256':verified.hash(),'bytes':len(blob),
            'bpp':8*len(blob)/TOTAL_UNIQUE,'matrix_fit':fit,'capture_counts':counts,
            'matrix_count':len(fit),'changed_groups':sum(x['changed_groups'] for x in fit.values()),
            'source_sha256':archive.hash(),'sealed_before_QA':True}
    save(out/'candidate_sealed_before_QA.json',sealed)
    with TemporaryDirectory(dir=out) as temp:
        root=Path(temp)
        original=root/'original.cb59';original.write_bytes(archive.blob)
        models={'FP32':teacher}
        for arm, path in [('CB59',original),('CB175',candidate)]:
            folder=root/arm;folder.mkdir()
            model,meta,_=make_model('baseline',SimpleNamespace(original=path,folder=folder))
            if not meta['zero_dense_pretrained_2D_hot_weights'] or any(p.ndim>=2 for p in model.parameters()):
                raise AssertionError('Dense weights in compact inference')
            if model.model.embed_tokens.handle is not model.lm_head.handle:
                raise AssertionError('Broken tied compact weights')
            models[arm]=model
        native_audit=matrix_roundtrip(verified,models['CB175'].lm_head)
        save(out/'native_roundtrip.json',native_audit)
        speed=matched_speed(models,tok,cases)
        save(out/'matched_native_speed.json',speed)
        print(json.dumps({'stage':'native_speed','median_ratio':speed['median_speed_ratio']}),flush=True)
        def progress(records,done):
            save(out/'answers_partial.json',{'completed_cases':done,'answers':records})
            print(json.dumps({'stage':'real_answers','completed_questions_per_arm':done}),flush=True)
        records=answers(models,tok,cases,progress)
        for arm in ('CB59','CB175'):
            if any(p.ndim>=2 for p in models[arm].parameters()):
                raise AssertionError('Dense trained matrix appeared after generation')
    if hashlib.sha256(candidate.read_bytes()).hexdigest()!=sealed['candidate_sha256']:
        raise AssertionError('Candidate changed during QA')
    result={'protocol':frozen,'candidate':sealed,'native_audit':native_audit,
            'native_speed':speed,'summary':summarize(records),'answers':records,
            'complete':True,'generations':sum(len(r) for r in records.values()),
            'ungraded_answers_require_text_review':True,
            'quality_recovery_not_inferred_from_calibration_loss':True}
    save(out/'v175_results.json',result)
    print(json.dumps({'stage':'complete','generations':result['generations'],
                      'summary':result['summary'],'bpp':sealed['bpp']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.source,a.out)
