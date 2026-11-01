"""Exact two-codec decision with one shared dictionary opening cost."""
from numbers import Integral


def nonnegative_integer(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError('byte costs must be nonnegative integers')
    return int(value)


def select_formats(base_bytes, dual_bytes, dictionary_bytes):
    base = tuple(map(nonnegative_integer, base_bytes))
    dual = tuple(map(nonnegative_integer, dual_bytes))
    opening = nonnegative_integer(dictionary_bytes)
    if len(base) != len(dual):
        raise ValueError('candidate counts differ')
    # Equality uses CL8: avoids an unnecessary decoder dependency.
    savings = sum(max(0, b - d) for b, d in zip(base, dual))
    use_dictionary = savings > opening
    modes = tuple('DUAL' if use_dictionary and d < b else 'CL8'
                  for b, d in zip(base, dual))
    payload = sum(d if m == 'DUAL' else b
                  for b, d, m in zip(base, dual, modes))
    return {'modes': modes, 'payload_bytes': payload,
            'dictionary_bytes': opening if use_dictionary else 0,
            'total_bytes': payload + (opening if use_dictionary else 0),
            'all_base_bytes': sum(base), 'gross_saving_bytes': savings}
