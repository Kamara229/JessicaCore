"""Research-only nested CALIBRI v88 multidimensional output-weight facade."""
def run(*args,**kwargs):
    from .pipeline import run as execute
    return execute(*args,**kwargs)
