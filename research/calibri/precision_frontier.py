"""Exact small-instance allocation under additive per-stratum error budgets.

This is a reference optimizer, not a scalable full-model solver. Errors must
be measured on calibration inputs, never on the held-out evaluation set.
Additivity holds for disjoint output-row pages of ONE linear operator using
unnormalized squared output error. It does not hold across a full network.
"""
from dataclasses import dataclass
import math
from shared_cost import nonnegative_integer


@dataclass(frozen=True)
class Candidate:
    name: str
    byte_count: int
    errors: tuple
    needs_dictionary: bool = False


@dataclass(frozen=True)
class State:
    byte_count: int
    errors: tuple
    dictionary_open: bool
    choices: tuple


def _dominates(a, b):
    return (a.dictionary_open == b.dictionary_open
            and a.byte_count <= b.byte_count
            and all(x <= y for x, y in zip(a.errors, b.errors)))


def allocate(pages, error_limits, dictionary_bytes=0, max_states=20000):
    limits = tuple(float(v) for v in error_limits)
    if not limits or any(not math.isfinite(v) or v < 0 for v in limits):
        raise ValueError('finite nonnegative error budgets required')
    opening = nonnegative_integer(dictionary_bytes)
    if nonnegative_integer(max_states) < 1:
        raise ValueError('max_states must be positive')
    frontier = [State(0, (0.,) * len(limits), False, ())]
    for page in pages:
        page = tuple(page)
        if not page or len({c.name for c in page}) != len(page):
            raise ValueError('nonempty page with unique candidate names required')
        for c in page:
            nonnegative_integer(c.byte_count)
            if (len(c.errors) != len(limits)
                    or any(not math.isfinite(v) or v < 0 for v in c.errors)):
                raise ValueError('invalid candidate errors')
        next_frontier = []
        for state in frontier:
            for c in page:
                errors = tuple(a + b for a, b in zip(state.errors, c.errors))
                if any(a > b for a, b in zip(errors, limits)):
                    continue
                cost = state.byte_count + c.byte_count
                if c.needs_dictionary and not state.dictionary_open:
                    cost += opening
                new = State(cost, errors, state.dictionary_open or c.needs_dictionary,
                            state.choices + (c.name,))
                if any(_dominates(old, new) for old in next_frontier):
                    continue
                next_frontier = [old for old in next_frontier
                                 if not _dominates(new, old)]
                next_frontier.append(new)
                if len(next_frontier) > max_states:
                    raise RuntimeError('exact frontier limit exceeded; no approximation returned')
        if not next_frontier:
            raise ValueError('no allocation meets all calibration error budgets')
        frontier = next_frontier
    return min(frontier, key=lambda s: (s.byte_count, sum(s.errors), s.choices))
