"""Exact-length microbatch planner for one existing logical optimizer batch.
No padding, no resampling, no extra optimizer steps, no changes to loss weights.
"""
from collections import defaultdict

def plan(rows, indices, max_batch):
    if max_batch<1 or not indices:raise ValueError("nonempty batch required")
    buckets=defaultdict(list)
    for occurrence,i in enumerate(indices):
        r=rows[i]
        if r.get("split","TRAIN")!="TRAIN":raise ValueError("TRAIN only")
        p,g=r["p"],r["g"]
        if len(p)<2 or not g or g[-1]!=2 or 2 in g[:-1] or len(p)+len(g)>=160:
            raise ValueError("invalid causal target sequence")
        for pos in range(len(g)):
            buckets[len(p)+pos].append((occurrence,i,pos))
    result=[]
    for length,items in sorted(buckets.items()):
        for j in range(0,len(items),max_batch):
            result.append((length,tuple(items[j:j+max_batch])))
    expected=sum(len(rows[i]["g"]) for i in indices)
    assert sum(len(items) for _,items in result)==expected
    assert len({(occ,pos) for _,items in result for occ,i,pos in items})==expected
    return result

def metrics(rows,indices,max_batch):
    groups=plan(rows,indices,max_batch)
    original=sum(len(rows[i]["g"]) for i in indices)
    return {"target_positions":original,"core_calls":len(groups),
            "batch_sizes":[len(x) for _,x in groups],
            "prefix_tokens":sum(length*len(x) for length,x in groups),
            "original_core_calls":original}
