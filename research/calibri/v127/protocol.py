"""Predeclared oracle-only diagnostic; gold is NEVER part of inference policy."""
SEED=127
STEPS=12
LEARNING_RATE=0.4
MAX_DELTA=.7
SHARED_GREEN_SHA="4a1f9ba4a0431569965ed9c0f786f7131b9426806ffd851f1b54ef8f6e585605"
MODEL_HOST_ACTIONS=41822331451
GREEN_ACTIONS=41828640085
ORACLE_CASES=(
 ("What is 27 plus 15? Reply with the number.","42","math"),
 ("What is 61 minus 24? Reply with the number.","37","math"),
 ("What is 8 multiplied by 7? Reply with the number.","56","math"),
 ("What is the capital of Croatia? Reply briefly.","Zagreb","geography"),
 ("What is the capital of Hungary? Reply briefly.","Budapest","geography"),
 ("Is Venus a star? Reply Yes or No.","No","boolean"),
)
TRAIN_MODES=("shared_six_oracle","individual_sixty_oracle")
assert len(ORACLE_CASES)==6 and len({q for q,g,d in ORACLE_CASES})==6
assert MAX_DELTA==.7 and STEPS==12
