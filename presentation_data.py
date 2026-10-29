"""Provenance checks for the notebook experiment shown to a judging panel."""
import hashlib
import json
import math
from pathlib import Path


def notebook_evaluation(root):
    root = Path(root)
    try:
        record = json.loads((root/'static'/'presentation-evaluation.json').read_text())
        if not isinstance(record, dict):
            return None
        notebook = root/'notebooks'/'GhostDemand_Training_Improved.ipynb'
        if hashlib.sha256(notebook.read_bytes()).hexdigest() != record['notebook_sha256']:
            return None
        sizes = record['split_sizes']
        if (not isinstance(sizes, dict) or set(sizes) != {'train', 'validation', 'test'}
                or any(type(value) is not int or value <= 0 for value in sizes.values())
                or sum(sizes.values()) != record['examples'] or record.get('synthetic') is not True):
            return None
        for key in ('macro_f1', 'accuracy', 'threshold', 'coverage'):
            value = record[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None
