# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Intel Corporation. All rights reserved.
# --------------------------------------------------------------------------
"""Shared ONNX external-data save helper for the Z-Image-Turbo exporters."""

import glob
import os
import re

import onnx
import onnx_ir as ir

INLINE_SIZE_THRESHOLD_BYTES = 1 * 1024**2
MAX_SHARD_SIZE_BYTES = int(1.9 * 1024**3)  # stays under the browser's 2 GiB ArrayBuffer ceiling

_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})")  # onnx_ir's "<stem>-000i-of-000N<ext>" naming


def save_ir_model_sharded(
    model, out_dir, onnx_filename, *, size_threshold_bytes, max_shard_size_bytes, callback=None
):
    """Save with small weights inline and external data sharded; returns the `.onnx` path."""
    out_path = os.path.join(out_dir, onnx_filename)
    data_base = onnx_filename + "_data"
    stem, ext = os.path.splitext(data_base)

    # onnx_ir refuses to overwrite a pre-existing shard file, so clear stale outputs first.
    stale = [out_path, *glob.glob(os.path.join(out_dir, data_base + "*")),
             *glob.glob(os.path.join(out_dir, f"{stem}-*-of-*{ext}"))]
    for path in stale:
        if os.path.exists(path):
            os.remove(path)

    ir.save(
        model, out_path, external_data=data_base,
        size_threshold_bytes=size_threshold_bytes, max_shard_size_bytes=max_shard_size_bytes,
        callback=callback,
    )

    temp_shards = sorted(glob.glob(os.path.join(out_dir, f"{stem}-*-of-*{ext}")))
    if not temp_shards:
        return out_path

    remap = {}
    for path in temp_shards:
        base = os.path.basename(path)
        idx = int(_SHARD_RE.search(base).group(1))
        final = data_base if idx == 1 else f"{data_base}_{idx - 1}"
        os.replace(path, os.path.join(out_dir, final))
        remap[base] = final

    proto = onnx.load(out_path, load_external_data=False)
    for init in proto.graph.initializer:
        for kv in init.external_data:
            if kv.key == "location" and kv.value in remap:
                kv.value = remap[kv.value]
    onnx.save(proto, out_path)
    return out_path
