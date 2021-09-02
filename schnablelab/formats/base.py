# -*- coding: UTF-8 -*-
from pathlib import Path
from itertools import groupby

def read_block(handle, signal):
    """
    borrow from haibao's jcvi package 
    Useful for reading block-like file formats. such file usually startswith
    some signal, and in-between the signals are a record
    """
    signal_len = len(signal)
    it = (x[1] for x in groupby(handle, 
        key=lambda row: row.strip()[:signal_len]==signal))
    found_signal = False
    for header in it:
        header = list(header)
        for h in header[:-1]:
            h = h.strip()
            if h[:signal_len] != signal:
                continue
            yield h, [] # header only, no contents
        header = header[-1].strip()
        if header[:signal_len] != signal:
            continue
        found_signal = True
        seq = [s.strip() for s in next(it)]
        yield header, seq
    if not found_signal:
        handle.seek(0)
        seq = list(s.strip() for s in handle)
        yield None, seq
