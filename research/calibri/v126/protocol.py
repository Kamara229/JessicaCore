"""One-time heldout third-host diagnostic without tuning any neural weights."""
MODEL="distilbert/distilgpt2"
REVISION="2290a62682d06624634c1f46a6ad5be0f47f38aa"
SHA_V120="81dd5ed9ee813843972cc837446fde908fc7d3dd9d2be3122efe7e2bd8ae96aa"
SHA_V124="0f6ab409fc0dbb02ea1a10ad5cae6796b87de8dfb0e8c50c55db51670387e57c"
SHA_V125="a15fd78f124f866a096de2e0a3a77d2d4584c98a9d326f37eaf9c8a5417b466b"
MAX_NEURAL_BYTES=10_000_000
MAX_NEW_TOKENS=24
FACTS=(
 ("What is the capital of Mexico? Answer briefly.","Mexico City"),
 ("What is the capital of Turkey? Answer briefly.","Ankara"),
 ("What is the capital of Belgium? Answer briefly.","Brussels"),
 ("What is the capital of Kenya? Answer briefly.","Nairobi"),
 ("What is the capital of Peru? Answer briefly.","Lima"),
 ("What is the capital of Serbia? Answer briefly.","Belgrade"),
 ("What is the capital of South Korea? Answer briefly.","Seoul"),
 ("What is the capital of the Netherlands? Answer briefly.","Amsterdam"),
)
MATH=tuple((
    f"What is {a} plus {b}? Reply with the number.",str(a+b))
    for a,b in ((110,17),(116,23),(119,9),(127,18),(133,12),(142,15),(150,16),(172,21)))
VALID=FACTS+MATH
assert len(VALID)==16 and len(set(q for q,a in VALID))==16
