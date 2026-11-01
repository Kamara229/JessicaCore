# CALIBRI session 7 — first pretrained weights
This is an **isolated research-only GitHub Actions experiment**, NOT a new
Jessica inference backend. No modifications to Render, Supabase, Skills, chat or APK.

Model: EleutherAI/pythia-14m, Apache-2.0, public model.safetensors.
The workflow records its immutable revision SHA and binary SHA256.
It examines all 12 FFN projection matrices (not attention/embeddings).

Experiments: reproduce a row-addressable lossless combinadic ternary codec,
serialize EXACT-LIFT with matching, measure 2-bit+zlib and full codec bytes,
verify full ternary-symbol roundtrip and matvec with asymmetric row scales.
The experiment's row-wise ternarization **is not QMoE GPTQ** and its
~88.5% target sparsity is deliberately imposed, not achieved by QMoE.
Float-to-ternary distortion is reported separately; the pilot does not
measure language-model perplexity or tokens/s.

Scientific gating: improve physical size only if the TOTAL includes metadata;
never equate smaller weights with better model quality or GPU speed. A lossless
lift cannot remove the error introduced by prior quantization.

Reproduction:
```sh
python -m pip install numpy==1.26.4 scipy==1.14.1 safetensors==0.4.5 huggingface_hub==0.27.1
PYTHONPATH=research/calibri python -m unittest discover -s research/calibri -v
PYTHONPATH=research/calibri python research/calibri/benchmark.py --output results/calibri_v07_pythia14m.json
```

Future gates: real activation calibration; whole-model perplexity; actual
Switch-base-8 MoE and official QMoE weights; fused CUDA/Triton benchmark.
