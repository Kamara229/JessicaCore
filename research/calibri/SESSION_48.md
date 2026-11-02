# CALIBRI v48 — cross-domain acceptance gate

Frozen method: fit both physical INT6 residual-row candidates using WikiText TRAIN-A only. Compare gradient-ranked and equal-byte module-matched SSE patches against plain INT6 on disjoint WikiText TRAIN-B and AG News TRAIN first 2048 tokens. Accept gradient only if NLL improves on both domains against both competitors by 1e-5; otherwise accept SSE if it improves on both against baseline; otherwise keep baseline.

Base physical model CB46: 10,803,661 bytes. Each candidate CR47 physical sidecar must have exactly the same byte size, count all per-row data and metadata, and meet 6.50 bpw including a 15,480-byte reserved correction budget. Holdout texts are not read before the gate decision: Yelp test[320:384] and ARC-Challenge validation[128:256], 2048 tokens each.

Benchmark selected versus original pretrained Pythia using a real 128-token prefill followed by 32 autoregressive KV-cache steps (two warmups, seven repetitions), with a target of less than 20% warm latency increase on each of two CPU runners. All PPL regressions must be recorded; failing the quality or speed goals is a scientific result rather than a checksum failure. Runtime decodes INT6 at startup to dense FP32; active RAM is not compressed. Research-only, no production deployment.
