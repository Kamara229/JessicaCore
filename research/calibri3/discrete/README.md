# V010 preparation only

No real model test or training authorized by this branch alone. V009 is still running. Pin the next physical checkpoint only after V009 result is reviewed; if it fails, retain V008/G43 and explicitly investigate learning failure first.

The proposal computes first-order loss changes from actual input activations and output gradients in row chunks. No dense weight or dense weight-gradient matrix is required. Local synthetic CPU check verifies exact linear-objective score, bounded distinct assignments, rejection and exception rollback; this is not model quality or GPU performance evidence.

Before real integration: ensure capture of each selected source's inputs and output gradients, use original TRAIN only with replay objective, clear all legacy and compact caches safely, measure physical packed delta bytes and bpp, restore original indices on failure, verify reloaded candidate and all earlier replay. Do not use DEV to choose proposals. No new model architecture or weights in this preparation.
