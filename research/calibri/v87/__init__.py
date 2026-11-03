"""Research-only CALIBRI v87 public facade; no runtime integration."""
def run(*a, **k):
    from .pipeline import run as execute
    return execute(*a, **k)
