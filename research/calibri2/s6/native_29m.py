"""S6 native C CPU: paired compact 29M S5 vs S6, identical ABI and prompts.
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
from calibri2.s6.train_29m import PARAMS, START_BYTES, START_SHA
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
        raise AssertionError("Native speed requires sealed exact S5 or S6 packed source")
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
    p.add_argument("s3", type=Path)
    p.add_argument("s4", type=Path)
    p.add_argument("s4_metadata", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    metadata = json.loads(a.s4_metadata.read_text())
    expected = metadata["result_sha256"]
    cfg = build_config(("grow_320x12", 320, 896, 12, 5, 1))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    ids = [tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False) for q in QUESTIONS]
    if not all(2 < len(x) < 120 for x in ids):
        raise AssertionError("Untrusted native speed prompt")
    binary = Path(__file__).resolve().parents[2]/"calibri"/"direct_kernel_v59.c"
    kernel = NativeKernel(compile_kernel(binary, a.out/"libcalibri2_s6_native.so"))
    s3, info3 = packed_model(a.s3, START_SHA, cfg, kernel)
    s4, info4 = packed_model(a.s4, expected, cfg, kernel)
    models = {"S5": s3, "S6": s4}
    for m in models.values():
        eight_real_tokens(m, ids[0])
    trials = []
    for repeat in range(3):
        for i, prompt_ids in enumerate(ids):
            timings = {}
            observed = {}
            names = ("S5", "S6") if (repeat + i) % 2 == 0 else ("S6", "S5")
            for name in names:
                timings[name], observed[name] = eight_real_tokens(models[name], prompt_ids)
            ratio = timings["S5"] / timings["S6"]
            trials.append({"repeat": repeat, "case": i, "seconds": timings,
                           "S6_over_S5": ratio, "real_generated_ids": observed})
    median = statistics.median(row["S6_over_S5"] for row in trials)
    result = {"stage": "S6_NATIVE_PAIRED_CPU", "s3_sha": START_SHA, "s4_sha": expected,
              "physical_bpp": 8*START_BYTES/PARAMS, "trials": trials,
              "median_S6_native_speed_relative_to_S5": median,
              "meets_90_percent": median >= .9, "native_C_contract_S5": info3,
              "native_C_contract_S6": info4, "tokens_per_trial": TOKEN_CAP,
              "no_dense_2D_inference_weight": True, "two_CPU_threads": True,
              "not_135M_baseline_comparison": True}
    save(a.out/"ACTUAL_CALIBRI2_S6_NATIVE_SPEED.json", result)
    print(json.dumps({"stage": "S6_NATIVE_PAIRED_CPU_DONE",
                      "trials": len(trials), "ratio": median, "pass_90pct": median >= .9,
                      "bpp": result["physical_bpp"]}), flush=True)

if __name__ == "__main__":
    main()
