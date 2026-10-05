#!/usr/bin/env python3
"""Hardlink the 2.75 EXL3NE tree, then rewrite the two shards that hold L2-49
o_proj / down_proj with the K5-encoded derisked tensors."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def collect_packed(k5: Path) -> dict[str, dict[str, torch.Tensor]]:
    """prefix -> {trellis,suh,svh,mul1}."""
    out: dict[str, dict[str, torch.Tensor]] = {}
    for f in sorted(k5.glob("L*.safetensors")):
        name = f.name  # L02.o_proj.safetensors
        stem = name[: -len(".safetensors")]
        L_s, kind = stem.split(".", 1)
        L = int(L_s[1:])
        if kind == "o_proj":
            prefix = f"model.layers.{L}.self_attn.o_proj"
        elif kind == "dense_down":
            prefix = f"model.layers.{L}.mlp.down_proj"
        elif kind == "shared_down":
            prefix = f"model.layers.{L}.mlp.shared_experts.down_proj"
        else:
            raise RuntimeError(kind)
        tensors = {}
        with safe_open(str(f), framework="pt") as st:
            for k in st.keys():
                tensors[k] = st.get_tensor(k)
        for need in ("trellis", "suh", "svh", "mul1"):
            if need not in tensors:
                raise RuntimeError(f"{f} missing {need}")
        out[prefix] = tensors
    return out


def rewrite_shard(src: Path, dst: Path, packed: dict[str, dict[str, torch.Tensor]]) -> dict:
    replaced = []
    tensors = {}
    with safe_open(str(src), framework="pt") as st:
        keys = list(st.keys())
        for k in keys:
            tensors[k] = st.get_tensor(k)
    for prefix, pt in packed.items():
        for suf, val in pt.items():
            key = f"{prefix}.{suf}"
            if key not in tensors:
                continue
            old = tensors[key]
            if tuple(old.shape) != tuple(val.shape):
                raise RuntimeError(f"{key} shape {tuple(val.shape)} != stock {tuple(old.shape)}")
            if old.dtype != val.dtype:
                # mul1 is I32 scalar; allow matching via .to
                val = val.to(old.dtype)
            tensors[key] = val.contiguous()
            replaced.append(key)
    tmp = dst.with_suffix(".safetensors.tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, dst)
    return {"shard": dst.name, "n_tensors": len(tensors), "replaced": replaced, "bytes": dst.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--k5", required=True)
    ap.add_argument("--shards", nargs="+", default=["model-00002-of-00080.safetensors", "model-00003-of-00080.safetensors"])
    args = ap.parse_args()
    stock, dest, k5 = Path(args.stock), Path(args.dest), Path(args.k5)
    packed = collect_packed(k5)
    print(f"packed prefixes: {len(packed)}", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    # hardlink every file, then unlink shards we will rewrite
    for p in stock.iterdir():
        if p.name.startswith("."):
            continue
        t = dest / p.name
        if t.exists() or t.is_symlink():
            continue
        if p.is_file():
            os.link(p, t)
        elif p.is_dir():
            shutil.copytree(p, t, copy_function=os.link, dirs_exist_ok=True)
    report = {"replaced_total": 0, "shards": []}
    for name in args.shards:
        src = stock / name
        dst = dest / name
        if dst.exists():
            # drop the hardlink so we do not clobber stock's inode
            if dst.stat().st_ino == src.stat().st_ino:
                dst.unlink()
            else:
                print(f"rewrite existing unique {name}", flush=True)
        print(f"rewrite {name}", flush=True)
        info = rewrite_shard(src, dst, packed)
        report["shards"].append(info)
        report["replaced_total"] += len(info["replaced"])
        print(f"  replaced {len(info['replaced'])} tensors, {info['bytes']/1e9:.2f}G", flush=True)
        missing = [f"{pfx}.{s}" for pfx in packed for s in packed[pfx]
                   if f"{pfx}.{s}" not in {k for sh in report["shards"] for k in sh["replaced"]}]
    # verify every packed tensor landed
    landed = {k for sh in report["shards"] for k in sh["replaced"]}
    missing = []
    for pfx, pt in packed.items():
        for s in pt:
            key = f"{pfx}.{s}"
            if key not in landed:
                missing.append(key)
    report["missing"] = missing
    (dest / "ABLIT_APPLY_REPORT.json").write_text(json.dumps(report, indent=2))
    note = {
        "parent": str(stock),
        "donor": "drowzeys/keys-GLM-5.3-EXL3-Abliterated",
        "method": "Blackfrost derisk o_proj + residual down_proj L2-49, re-quant EXL3 K5 mul1",
        "layers": "2-49",
        "anchors_untouched": "0-1 and 50-78 including MTP",
        "routed_experts": "untouched (same as donor bake)",
        "n_prefixes": len(packed),
        "replaced_tensors": report["replaced_total"],
        "missing": missing,
    }
    (dest / "ABLIT.json").write_text(json.dumps(note, indent=2) + "\n")
    if missing:
        print("MISSING", missing, file=sys.stderr)
        return 1
    print("SPLICE_OK", report["replaced_total"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
