"""Loads configs/default.yaml. The only place configuration is read."""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _mini_yaml(text: str):
    """Tiny YAML subset parser (dicts, scalars, inline lists, comments).

    Avoids a PyYAML dependency for a file we fully control.
    """
    root, stack = {}, [(-1, {})]
    stack[0][1].update(root)
    cur = {}
    root = cur
    stack = [(-1, cur)]
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip().split(" #")[0].rstrip()
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            node = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _scalar(val)
    return root

def _scalar(v: str):
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_scalar(x.strip()) for x in inner.split(",")] if inner else []
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v.strip('"').strip("'")

class Config(dict):
    """Attribute-accessible nested config."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(f"missing config key: {k}") from e
        return Config(v) if isinstance(v, dict) else v

    def path(self, key_path: str):
        node = self
        for part in key_path.split("."):
            node = node[part]
        return ROOT / node

REQUIRED = ["seed", "data", "corpus", "splits", "retrieval", "reader",
            "reliability", "evaluation"]

def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    cfg = Config(_mini_yaml(p.read_text()))
    for key in REQUIRED:
        if key not in cfg:
            raise KeyError(f"config is missing required section: {key!r}")
    s = cfg["splits"]
    total = sum(s[k] for k in ("reader_train", "rel_train", "rel_calib", "test"))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    return cfg

def dump(cfg: Config) -> str:
    return json.dumps(cfg, indent=2, sort_keys=True)
