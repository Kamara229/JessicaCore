"""Fixed-byte full-covariance feedback experiment; no answer parity promised."""
SOURCE_SHA = "50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
SOURCE_BYTES = 11229303
TOTAL_UNIQUE = 134515008
GROUP = 8
CENTERS = 32
DAMPING = 0.05
# The tied tensor serves input embedding as well as output projection.
# Preserve an isotropic source penalty in addition to head input covariance.
TIED_EMBEDDING_PENALTY = 1.0
FIT_CORPORA = ("wiki_fit", "news_fit")
MAX_NEW_TOKENS = 64
ARMS = ("FP32", "CB59", "CB175")
SPEED_CASE_IDS = (0, 16, 64, 80)


def calibration_key(path):
    # q/k/v and gate/up receive identical tensors; capture each once.
    path = path.removesuffix(".weight")
    if path == "model.embed_tokens":
        return "lm_head"
    if path.endswith((".k_proj", ".v_proj")):
        return path.rsplit(".", 1)[0] + ".q_proj"
    if path.endswith(".up_proj"):
        return path.rsplit(".", 1)[0] + ".gate_proj"
    return path
