"""Freeze source hashes and diagnostic probes before host transfer."""
GREEN_SHA="4a1f9ba4a0431569965ed9c0f786f7131b9426806ffd851f1b54ef8f6e585605"
GREEN_SOURCE_RUN=41828640085
HOST135_SOURCE_RUN=41822331451
PYTHIA="EleutherAI/pythia-14m"
PYTHIA_SHA="cf967c0a9a04383db6f7b1108d86b2962634b4ac"
PROBES=(
    "What is 4 plus 5? Answer:",
    "The capital of Norway is",
    "The little brown bird landed",
    "A scientist explains why rain happens:",
)
MAX_NEW_TOKENS=24
STEERING_MAGNITUDE=.15
