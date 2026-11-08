"""S20 native C CPU: paired compact 36M S7 vs S20, identical ABI and prompts.
Fixed 8-token work, two CPU threads, no dense inference matrices.
This is a matched SAME-ARCHITECTURE speed regression, NOT 135M FP32 parity.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, time
from pathlib import Path
import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from direct_runtime_v59 import compile_kernel, NativeKernel, bind_into_llama
from calibri2.s0.bootstrap import CANDIDATES, build_config
from calibri2.s8.broad_replay import SOURCE_PARAMS as PARAMS, SOURCE_BYTES as START_BYTES, SOURCE_SHA as START_SHA, MODEL_SPEC
from v174.run import save

QUESTIONS = (
 "Is 321 greater than 312? Reply with only Yes or No.",
 "In a story, Nara has a red folder. What two words describe it?",
 "What are the two secret words 'quiet harbor'?",
)
TOKEN_CAP = 8

def packed_model(path, expected_sha, cfg, kernel):
    raw = path.read_bytes()
    if len(raw) != START_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise AssertionError("Native speed requires sealed exact S7 or S20 packed source")
    archive = CB59(raw)
    if archive.meta["original_unique_parameter_count"] != PARAMS or 8 * len(raw) / PARAMS > .700:
        raise AssertionError("Unpaid physical model")
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32).eval()
    info = bind_into_llama(model, archive, kernel)
    if info["runtime_dense_multi_dim_source_parameters"] != 0:
        raise AssertionError("Dense FP32 matrix leaked into inference")
    return model, info

@torch.inference_mode()
def eight_real_tokens(model, ids):
    ctx = torch.tensor([ids], dtype=torch.long)
    generated = []
    start = time.perf_counter()
    for _ in range(TOKEN_CAP):
        logits = model(input_ids=ctx, use_cache=False).logits[0, -1]
        if not torch.isfinite(logits).all():
            raise AssertionError("Nonfinite native logits")
        nxt = int(logits.argmax())
        generated.append(nxt)
        ctx = torch.cat((ctx, ctx.new_tensor([[nxt]])), dim=1)
    return time.perf_counter() - start, generated

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source_archive", type=Path)
    p.add_argument("trained_archive", type=Path)
    p.add_argument("train_metadata", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    metadata = json.loads(a.train_metadata.read_text())
    expected = metadata["physical_final_SHA256"]
    cfg = build_config(MODEL_SPEC)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    ids = [tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False) for q in QUESTIONS]
    if not all(2 < len(x) < 120 for x in ids):
        raise AssertionError("Untrusted native speed prompt")
    binary = Path(__file__).resolve().parents[2]/"calibri"/"direct_kernel_v59.c"
    kernel = NativeKernel(compile_kernel(binary, a.out/"libcalibri2_s20_native.so"))
    s3, info3 = packed_model(a.source_archive, START_SHA, cfg, kernel)
    s4, info4 = packed_model(a.trained_archive, expected, cfg, kernel)
    models = {"S7": s3, "S20": s4}
    for m in models.values():
        eight_real_tokens(m, ids[0])
    trials = []
    for repeat in range(3):
        for i, prompt_ids in enumerate(ids):
            timings = {}
            observed = {}
            names = ("S7", "S20") if (repeat + i) % 2 == 0 else ("S20", "S7")
            for name in names:
                timings[name], observed[name] = eight_real_tokens(models[name], prompt_ids)
            ratio = timings["S7"] / timings["S20"]
            trials.append({"repeat": repeat, "case": i, "seconds": timings,
                           "S20_over_S7": ratio, "real_generated_ids": observed})
    median = statistics.median(row["S20_over_S7"] for row in trials)
    result = {"stage": "S20_NATIVE_PAIRED_CPU", "s3_sha": START_SHA, "s4_sha": expected,
              "physical_bpp": 8*START_BYTES/PARAMS, "trials": trials,
              "median_S20_native_speed_relative_to_S7": median,
              "meets_90_percent": median >= .9, "native_C_contract_S7": info3,
              "native_C_contract_S20": info4, "tokens_per_trial": TOKEN_CAP,
              "no_dense_2D_inference_weight": True, "two_CPU_threads": True,
              "not_135M_baseline_comparison": True}
    save(a.out/"ACTUAL_CALIBRI2_S20_NATIVE_SPEED.json", result)
    print(json.dumps({"stage": "S20_NATIVE_PAIRED_CPU_DONE",
                      "trials": len(trials), "ratio": median, "pass_90pct": median >= .9,
                      "bpp": result["physical_bpp"]}), flush=True)

if __name__ == "__main__":
    main()
