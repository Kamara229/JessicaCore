"""Frozen CALIBRI v89 independent QA/KL/generation test facade."""
def run(*args,**kwargs):
    from .pipeline import run as execute
    return execute(*args,**kwargs)
