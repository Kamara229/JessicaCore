import argparse
from pathlib import Path
from .architecture import run
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("g05","f08","parent","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.out)
