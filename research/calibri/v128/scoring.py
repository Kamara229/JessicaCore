"""Strict short answers. Keep signs, decimal separators and chemical symbol case."""
def matches(text, gold, question=""):
    def clean(s):
        s = " ".join(s.strip().replace("−", "-").split())
        return s.removesuffix(".").strip()
    response, reference = clean(text), clean(gold)
    if "chemical symbol" in question.casefold():
        return response == reference
    return bool(reference) and response.casefold() == reference.casefold()
