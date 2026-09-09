# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Intel Corporation. All rights reserved.
# --------------------------------------------------------------------------
"""Standalone exporter for the Z-Image-Turbo transformer trunk (`ZImageTransformer2DModel`)."""

import argparse
import json
import math
import os
import types
from typing import Any

import onnx_ir as ir
import torch
from onnxruntime_genai.models.builders.base import Model

from external_data_utils import (
    INLINE_SIZE_THRESHOLD_BYTES,
    MAX_SHARD_SIZE_BYTES,
    save_ir_model_sharded,
)

PRECISION_CONFIGS = {
    "f16": {"builder_precision": "fp16", "int4_quant": False},
    "f32": {"builder_precision": "fp32", "int4_quant": False},
    "f16_int4_quant": {"builder_precision": "int4", "int4_quant": True},
    "f32_int4_quant": {"builder_precision": "int4", "int4_quant": True},
}

FILENAME_SUFFIXES = {
    "f16": "f16",
    "f32": "f32",
    "f16_int4_quant": "q4f16",
    "f32_int4_quant": "q4f32",
}

DEFAULT_OUTPUT_DIR = "z-image-turbo-onnx"


# Vendored from onnxruntime_genai.models.builder.
def set_io_dtype(precision, execution_provider, extra_options) -> ir.DataType:
    """Set the input/output precision of the ONNX model based on the provided precision and execution provider."""
    cpu_quant = precision in {"int4", "int8"} and execution_provider == "cpu"
    fp32_webgpu = execution_provider == "webgpu" and extra_options.get("use_webgpu_fp32", False)
    bf16_cuda = precision == "int4" and execution_provider in {"cuda", "trt-rtx"} and extra_options.get("use_cuda_bf16", False)

    if precision == "fp32" or cpu_quant or fp32_webgpu:
        return ir.DataType.FLOAT
    if precision == "bf16" or bf16_cuda:
        return ir.DataType.BFLOAT16
    return ir.DataType.FLOAT16


def set_onnx_dtype(precision: str, extra_options: dict[str, Any]) -> ir.DataType:
    """Set the ONNX model's internal precision based on the provided precision and extra options."""
    if precision == "int4":
        return ir.DataType.INT4 if extra_options.get("is_symmetric", True) else ir.DataType.UINT4
    if precision == "int8":
        return ir.DataType.INT8 if extra_options.get("is_symmetric", True) else ir.DataType.UINT8
    to_onnx_dtype = {
        "fp32": ir.DataType.FLOAT,
        "fp16": ir.DataType.FLOAT16,
        "bf16": ir.DataType.BFLOAT16,
    }
    return to_onnx_dtype[precision]


def load_diffusers_config(input_path):
    """Load a diffusers-style `config.json` (e.g. Z-Image-Turbo's `transformer/config.json`)."""
    config_path = os.path.join(input_path, "config.json")
    with open(config_path) as f:
        raw_config = json.load(f)
    raw_config.setdefault("_name_or_path", input_path)
    return types.SimpleNamespace(**raw_config)


def parse_extra_options(pairs):
    """Parse `key=value` strings (as passed via `--extra_options`) into a dict."""
    extra_options = {}
    for kv_str in pairs or []:
        key, _, value = kv_str.partition("=")
        extra_options[key.strip()] = value.strip()

    if "op_types_to_quantize" in extra_options:
        extra_options["op_types_to_quantize"] = tuple(extra_options["op_types_to_quantize"].split("/"))

    for bool_key in ("hf_remote", "use_webgpu_fp32", "is_symmetric", "use_qdq"):
        if bool_key in extra_options:
            extra_options[bool_key] = extra_options[bool_key] not in ("false", "False", "0")

    for int_key in ("block_size", "accuracy_level"):
        if int_key in extra_options:
            extra_options[int_key] = int(extra_options[int_key])

    return extra_options


class ZImageTransformerModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Translates the diffusers config into the causal-LM-shaped namespace Model.__init__
        # expects; everything LLM-specific it sets up is unused by this class.
        fake_config = types.SimpleNamespace(
            _name_or_path=getattr(config, "_name_or_path", "z-image-transformer"),
            architectures=["ZImageTransformer2DModel"],
            hidden_size=config.dim,
            num_attention_heads=config.n_heads,
            num_key_value_heads=config.n_kv_heads,
            num_hidden_layers=config.n_layers,
            intermediate_size=config.dim,
            vocab_size=0,
            hidden_act="silu",
            max_position_embeddings=max(config.axes_lens),
            rms_norm_eps=config.norm_eps,
        )
        super().__init__(fake_config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.model_type = "z-image-transformer"
        # Bidirectional attention everywhere in this model (no causal mask, no KV cache).
        self.attention_attrs["unidirectional"] = False

        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_layers = config.n_layers
        self.n_refiner_layers = config.n_refiner_layers
        self.in_channels = config.in_channels
        self.cap_feat_dim = config.cap_feat_dim
        self.norm_eps = config.norm_eps
        self.qk_norm = config.qk_norm
        self.rope_theta = config.rope_theta
        self.t_scale = config.t_scale
        self.axes_dims = list(config.axes_dims)
        self.axes_lens = list(config.axes_lens)
        self.patch_size = config.all_patch_size[0]
        self.f_patch_size = config.all_f_patch_size[0]
        self.patch_key = f"{self.patch_size}-{self.f_patch_size}"

        assert self.head_size == sum(self.axes_dims), (
            f"head_size ({self.head_size}) must equal sum(axes_dims) ({sum(self.axes_dims)})"
        )

        # TimestepEmbedder / AdaLN embedding width (diffusers: `ADALN_EMBED_DIM = 256`).
        self.adaln_embed_dim = min(self.dim, 256)
        self.t_freq_dim = 256
        self.t_mid_dim = 1024
        self.patch_dim = self.f_patch_size * self.patch_size * self.patch_size * self.in_channels

        # float16 overflow workaround, exact powers of two, RMSNorm-scale-invariant.
        self.pre_out_proj_scale = 1.0 / 128.0
        self.ff_gate_scale = 1.0 / 8.0
        self.ff_up_scale = 1.0 / 16.0

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(self, input_path):
        from diffusers import ZImageTransformer2DModel

        model = ZImageTransformer2DModel.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
        )
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_model(self, out_dir):
        """Save with small weights inline and sharded external data (overrides Model.save_model)."""
        from tqdm import tqdm

        print(f"Saving ONNX model in {out_dir}")
        already_quantized_in_qdq_format = self.quant_type is not None and self.quant_attrs["use_qdq"]
        if self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4, ir.DataType.INT8, ir.DataType.UINT8} and not already_quantized_in_qdq_format:
            model = self.to_nbits()
        else:
            model = self.model
        model.graph.sort()

        with tqdm() as pbar:
            total_set = False

            def callback(tensor, metadata):
                nonlocal total_set
                if not total_set:
                    pbar.total = metadata.total
                    total_set = True
                pbar.update()
                pbar.set_description(f"Saving {tensor.name} ({tensor.dtype.short_name()}, {tensor.shape})")

            save_ir_model_sharded(
                model, out_dir, self.filename,
                size_threshold_bytes=INLINE_SIZE_THRESHOLD_BYTES,
                max_shard_size_bytes=MAX_SHARD_SIZE_BYTES,
                callback=callback,
            )

        if not os.listdir(self.cache_dir):
            os.rmdir(self.cache_dir)

    # ------------------------------------------------------------------
    # Inputs / outputs
    # ------------------------------------------------------------------
    def make_inputs_and_outputs(self):
        self.input_names = {
            "hidden_states": "hidden_states",
            "encoder_hidden_states": "encoder_hidden_states",
            "timestep": "timestep",
        }
        self.output_names = {"sample": "sample"}

        hidden_states = self.make_value(
            "hidden_states", self.io_dtype, shape=[1, self.in_channels, "height", "width"]
        )
        encoder_hidden_states = self.make_value(
            "encoder_hidden_states", self.io_dtype, shape=[1, "cap_seq_len", self.cap_feat_dim]
        )
        timestep = self.make_value("timestep", self.io_dtype, shape=[1])
        self.model.graph.inputs.extend([hidden_states, encoder_hidden_states, timestep])

        sample = self.make_value("sample", self.io_dtype, shape=[1, self.in_channels, "height", "width"])
        self.model.graph.outputs.append(sample)

    # ------------------------------------------------------------------
    # Small node-building helpers (thin, name-returning wrappers around the
    # low-level builders inherited from `Model`)
    # ------------------------------------------------------------------
    def _const(self, dtype: ir.DataType, value) -> str:
        """Return the name of an auto-created scalar/list constant (see `Model.make_constant`)."""
        return f"/model/constants/{self.to_str_dtype(dtype)}/{value}"

    def _linear(self, name, root_input, linear, shape, seq_dim=None, exclude_from_quant=False):
        """Apply an `nn.Linear` (weight + optional bias) to `root_input`, returning the output name.

        seq_dim must be unique per logically-distinct sequence, not the shared default.
        """
        if seq_dim is None:
            seq_dim = shape[1] if len(shape) == 3 and isinstance(shape[1], str) else f"dim1_of_{name}"
        if exclude_from_quant:
            # Registers `name` so genai 0.15.2's deferred int4 quantization (to_nbits) skips this weight.
            self.quant_attrs["nodes_to_exclude"].append(name)
        self.make_matmul(linear, name, root_input, seq_dim=seq_dim)
        output = f"{name}/output_0"
        # Re-stamp: make_matmul assumes the generic 3D shape, which doesn't always match.
        self.make_value(output, self.io_dtype, shape=shape)
        if linear.bias is not None:
            self.make_add_bias(linear.bias, f"{name}/Add", output, seq_dim=seq_dim)
            output = f"{name}/Add/output_0"
            self.make_value(output, self.io_dtype, shape=shape)
        return output

    def _rms_norm(self, name, root_input, weight, shape, eps=None):
        weight_name = name[1:].replace("/", ".") + ".weight"
        self.make_initializer(weight, weight_name, to=self.io_dtype)
        output = f"{name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, weight_name],
            outputs=[output],
            name=name,
            axis=-1,
            epsilon=eps if eps is not None else self.norm_eps,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape=shape)
        return output

    def _silu(self, name, root_input, shape):
        sigmoid_name = f"{name}/Sigmoid"
        self.make_sigmoid(sigmoid_name, root_input, self.io_dtype, shape)
        self.make_mul(name, [root_input, f"{sigmoid_name}/output_0"], self.io_dtype, shape)
        return f"{name}/output_0"

    def _mul(self, name, inputs, shape):
        self.make_mul(name, inputs, self.io_dtype, shape)
        return f"{name}/output_0"

    def _add(self, name, inputs, shape):
        self.make_add(name, inputs, self.io_dtype, shape)
        return f"{name}/output_0"

    def _add_scalar(self, name, root_input, value, shape):
        self.make_add(name, [root_input, self._const(self.io_dtype, value)], self.io_dtype, shape)
        return f"{name}/output_0"

    def _unsqueeze(self, name, root_input, axes, shape, dtype=None):
        axes_name = self._const(ir.DataType.INT64, list(axes))
        self.make_unsqueeze(name, [root_input, axes_name], dtype or self.io_dtype, shape)
        return f"{name}/output_0"

    def _reshape(self, name, root_input, shape_parts, out_shape, dtype=None):
        """Reshape `root_input`; each of `shape_parts` is a Python int or a 1D INT64 tensor name."""
        dtype = dtype or self.io_dtype
        parts = []
        for part in shape_parts:
            if isinstance(part, int):
                parts.append(self._const(ir.DataType.INT64, [part]))
            else:
                parts.append(part)
        if len(parts) == 1:
            shape_tensor = parts[0]
        else:
            shape_concat_name = f"{name}/ShapeConcat"
            self.make_concat(shape_concat_name, parts, ir.DataType.INT64, shape=[len(parts)], axis=0)
            shape_tensor = f"{shape_concat_name}/output_0"
        self.make_reshape(name, [root_input, shape_tensor], dtype, out_shape)
        return f"{name}/output_0"

    def _dim_of(self, name, root_input, axis, full_shape_len):
        """Return the name of a 1D INT64 tensor of length 1 holding `root_input`'s size along `axis`."""
        shape_name = f"{name}/Shape"
        self.make_shape(shape_name, root_input, shape=[full_shape_len])
        gather_name = f"{name}/Gather"
        self.make_gather(
            gather_name, [f"{shape_name}/output_0", self._const(ir.DataType.INT64, [axis])],
            dtype=ir.DataType.INT64, shape=[1], axis=0,
        )
        return f"{gather_name}/output_0"

    def _scalar(self, name, dim1_tensor):
        """Squeeze a 1D length-1 INT64 tensor down to a 0-D scalar (needed by `Range`)."""
        self.make_squeeze(name, [dim1_tensor, self._const(ir.DataType.INT64, [0])], ir.DataType.INT64, shape=[])
        return f"{name}/output_0"

    def _int_scalar_add(self, name, a, b):
        """Add two 0-D INT64 scalars."""
        self.make_add(name, [a, b], ir.DataType.INT64, shape=[])
        return f"{name}/output_0"

    def _int_div(self, name, dim1_tensor, divisor):
        self.make_div(name, [dim1_tensor, self._const(ir.DataType.INT64, [divisor])], ir.DataType.INT64, shape=[1])
        return f"{name}/output_0"

    def _int_mul(self, name, a, b):
        self.make_mul(name, [a, b], ir.DataType.INT64, shape=[1])
        return f"{name}/output_0"

    # ------------------------------------------------------------------
    # RoPE: precomputed per-axis frequency tables + dynamic position ids
    # ------------------------------------------------------------------
    def _make_rope_tables(self):
        """Precompute the 3 per-axis (cos, sin) frequency tables, [length, dim//2, 2]."""
        self.rope_table_names = []
        for i, (dim, length) in enumerate(zip(self.axes_dims, self.axes_lens)):
            freqs = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            positions = torch.arange(length, dtype=torch.float64)
            angles = torch.outer(positions, freqs).float()  # [length, dim//2]
            cos_vals = torch.cos(angles)  # [length, dim//2]
            sin_vals = torch.sin(angles)  # [length, dim//2]
            table = torch.stack([cos_vals, sin_vals], dim=-1)  # [length, dim//2, 2]
            name = f"model.rope.axis_{i}_freqs"
            self.make_initializer(table, name, to=self.io_dtype)
            self.rope_table_names.append(name)

    def _gather_axis_freqs(self, name, axis_idx, position_ids, num_tokens_shape):
        dim = self.axes_dims[axis_idx]
        self.make_gather(
            name, [self.rope_table_names[axis_idx], position_ids], dtype=self.io_dtype,
            shape=[num_tokens_shape, dim // 2, 2], axis=0,
        )
        return f"{name}/output_0"

    def _build_freqs_cis(self, name, axis0_ids, axis1_ids, axis2_ids, num_tokens_shape):
        """Gather + concat the 3 axes into a per-token `[num_tokens, head_size//2, 2]` (cos, sin) tensor."""
        parts = [
            self._gather_axis_freqs(f"{name}/axis_{i}/Gather", i, ids, num_tokens_shape)
            for i, ids in enumerate((axis0_ids, axis1_ids, axis2_ids))
        ]
        self.make_concat(name, parts, self.io_dtype, shape=[num_tokens_shape, self.head_size // 2, 2], axis=1)
        return f"{name}/output_0"

    def _cos_sin_caches_from_freqs_cis(self, name, freqs_cis, num_tokens_shape):
        """Split `[tokens, head_size//2, 2]` into 2D cos/sin caches `[tokens, head_size//2]`."""
        half = self.head_size // 2
        cos_cache = f"{name}/Cos"
        sin_cache = f"{name}/Sin"
        self.make_gather(cos_cache, [freqs_cis, self._const(ir.DataType.INT64, 0)], dtype=self.io_dtype, shape=[num_tokens_shape, half], axis=2)
        self.make_gather(sin_cache, [freqs_cis, self._const(ir.DataType.INT64, 1)], dtype=self.io_dtype, shape=[num_tokens_shape, half], axis=2)
        return f"{cos_cache}/output_0", f"{sin_cache}/output_0"

    def _identity_position_ids(self, name, seq_len_scalar, out_dim_name):
        """Build identity `position_ids` `[1, seq] = [[0, 1, ..., seq-1]]` (int64)."""
        self.make_range(
            name, [self._const(ir.DataType.INT64, 0), seq_len_scalar, self._const(ir.DataType.INT64, 1)],
            ir.DataType.INT64, shape=[out_dim_name],
        )
        return self._unsqueeze(f"{name}/Unsqueeze", f"{name}/output_0", [0], [1, out_dim_name], dtype=ir.DataType.INT64)

    def _apply_rope(self, name, x, num_tokens_shape, num_heads, cos_cache, sin_cache, position_ids):
        """Apply interleaved-pair RoPE via `com.microsoft.RotaryEmbedding`; x: [1,tokens,num_heads,head_size]."""
        hidden = num_heads * self.head_size
        shape3 = [1, num_tokens_shape, hidden]
        x3d = self._reshape(f"{name}/Reshape3D", x, [1, -1, hidden], shape3)
        output = f"{name}/output_0"
        self.make_node(
            "RotaryEmbedding",
            inputs=[x3d, position_ids, cos_cache, sin_cache],
            outputs=[output],
            name=name,
            domain="com.microsoft",
            interleaved=1,
            num_heads=num_heads,
        )
        self.make_value(output, self.io_dtype, shape=shape3)
        return output

    def _build_position_grids(self, height, width, cap_seq_len):
        """Build dynamic position-id tensors for the image and caption token grids."""
        h_tok = self._int_div("/model/z_image/h_tokens", height, self.patch_size)
        w_tok = self._int_div("/model/z_image/w_tokens", width, self.patch_size)
        num_img_tokens = self._int_mul("/model/z_image/num_img_tokens", h_tok, w_tok)

        cap_len_scalar = self._scalar("/model/z_image/cap_len_scalar", cap_seq_len)
        cap_len_plus1 = self._int_scalar_add(
            "/model/z_image/cap_len_plus1", cap_len_scalar, self._const(ir.DataType.INT64, 1)
        )

        # --- caption position ids ---
        self.make_range(
            "/model/z_image/cap_axis0",
            [self._const(ir.DataType.INT64, 1), cap_len_plus1, self._const(ir.DataType.INT64, 1)],
            ir.DataType.INT64, shape=["cap_seq_len"],
        )
        cap_axis0 = "/model/z_image/cap_axis0/output_0"
        cap_axis12 = self._zeros_like_1d("/model/z_image/cap_axis12", cap_seq_len, "cap_seq_len")

        # --- image position ids ---
        img_axis0 = self._expand_scalar("/model/z_image/img_axis0", cap_len_plus1, num_img_tokens, "img_seq_len")

        h_tok_scalar = self._scalar("/model/z_image/h_tok_scalar", h_tok)
        w_tok_scalar = self._scalar("/model/z_image/w_tok_scalar", w_tok)
        self.make_range("/model/z_image/row_ids", [self._const(ir.DataType.INT64, 0), h_tok_scalar, self._const(ir.DataType.INT64, 1)], ir.DataType.INT64, shape=["h_tokens"])
        self.make_range("/model/z_image/col_ids", [self._const(ir.DataType.INT64, 0), w_tok_scalar, self._const(ir.DataType.INT64, 1)], ir.DataType.INT64, shape=["w_tokens"])
        row_ids = self._unsqueeze("/model/z_image/row_ids_u", "/model/z_image/row_ids/output_0", [1], ["h_tokens", 1], dtype=ir.DataType.INT64)
        col_ids = self._unsqueeze("/model/z_image/col_ids_u", "/model/z_image/col_ids/output_0", [0], [1, "w_tokens"], dtype=ir.DataType.INT64)
        grid_shape = self._concat_shape("/model/z_image/grid_shape", [h_tok, w_tok])
        row_grid = self._expand("/model/z_image/row_grid", row_ids, grid_shape, ["h_tokens", "w_tokens"])
        col_grid = self._expand("/model/z_image/col_grid", col_ids, grid_shape, ["h_tokens", "w_tokens"])

        img_axis1 = self._reshape("/model/z_image/img_axis1", row_grid, [num_img_tokens], ["img_seq_len"], dtype=ir.DataType.INT64)
        img_axis2 = self._reshape("/model/z_image/img_axis2", col_grid, [num_img_tokens], ["img_seq_len"], dtype=ir.DataType.INT64)

        return {
            "h_tok": h_tok,
            "w_tok": w_tok,
            "num_img_tokens": num_img_tokens,
            "cap_axis": (cap_axis0, cap_axis12, cap_axis12),
            "img_axis": (img_axis0, img_axis1, img_axis2),
        }

    def _zeros_like_1d(self, name, dim1_tensor, out_dim_name):
        scalar = self._scalar(f"{name}/scalar", dim1_tensor)
        shape_1d = self._unsqueeze(f"{name}/shape", scalar, [0], [1], dtype=ir.DataType.INT64)
        self.make_constant_of_shape(name, shape_1d, ir.tensor([0], dtype=ir.DataType.INT64), ir.DataType.INT64, shape=[out_dim_name])
        return f"{name}/output_0"

    def _expand_scalar(self, name, scalar_value, count_dim1_tensor, out_dim_name):
        """Broadcast a 0-D scalar to a 1D tensor whose length is given by `count_dim1_tensor`'s value."""
        value_1d = self._unsqueeze(f"{name}/Value", scalar_value, [0], [1], dtype=ir.DataType.INT64)
        self.make_expand(name, [value_1d, count_dim1_tensor], ir.DataType.INT64, shape=[out_dim_name])
        return f"{name}/output_0"

    def _concat_shape(self, name, dim1_tensors):
        self.make_concat(name, list(dim1_tensors), ir.DataType.INT64, shape=[len(dim1_tensors)], axis=0)
        return f"{name}/output_0"

    def _expand(self, name, root_input, shape_tensor, out_shape):
        self.make_expand(name, [root_input, shape_tensor], ir.DataType.INT64, shape=out_shape)
        return f"{name}/output_0"

    # ------------------------------------------------------------------
    # Timestep embedding
    # ------------------------------------------------------------------
    def _make_timestep_freqs(self):
        half = self.t_freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / half)
        self.make_initializer(freqs, "model.t_embedder.freqs", to=self.io_dtype)

    def _make_timestep_embedding(self, timestep, weights):
        half = self.t_freq_dim // 2
        t_scaled = self._mul("/model/t_embedder/Scale", [timestep, self._const(self.io_dtype, self.t_scale)], [1])
        t_unsq = self._unsqueeze("/model/t_embedder/Unsqueeze", t_scaled, [1], [1, 1])
        args = self._mul("/model/t_embedder/Args", [t_unsq, "model.t_embedder.freqs"], [1, half])
        cos_name = "/model/t_embedder/Cos"
        sin_name = "/model/t_embedder/Sin"
        self.make_cos(cos_name, args, self.io_dtype, [1, half])
        self.make_sin(sin_name, args, self.io_dtype, [1, half])
        freq_embed_name = "/model/t_embedder/Concat"
        self.make_concat(freq_embed_name, [f"{cos_name}/output_0", f"{sin_name}/output_0"], self.io_dtype, shape=[1, self.t_freq_dim], axis=-1)
        freq_embed = f"{freq_embed_name}/output_0"

        hidden = self._linear(
            "/model/t_embedder/mlp.0", freq_embed, weights.mlp[0], [1, self.t_mid_dim], exclude_from_quant=True
        )
        hidden = self._silu("/model/t_embedder/mlp.0/Silu", hidden, [1, self.t_mid_dim])
        adaln_input = self._linear(
            "/model/t_embedder/mlp.2", hidden, weights.mlp[2], [1, self.adaln_embed_dim], exclude_from_quant=True
        )
        return adaln_input

    # ------------------------------------------------------------------
    # AdaLN modulation
    # ------------------------------------------------------------------
    def _make_adaln(self, name, adaln_input, linear, dim):
        mod = self._linear(f"{name}/Linear", adaln_input, linear, [1, 4 * dim], exclude_from_quant=True)
        split_sizes = self._const(ir.DataType.INT64, [dim, dim, dim, dim])
        outs = [f"{name}/Split/output_{i}" for i in range(4)]
        self.make_split(f"{name}/Split", [mod, split_sizes], outs, [self.io_dtype] * 4, [[1, dim]] * 4, axis=-1)
        scale_msa_raw, gate_msa_raw, scale_mlp_raw, gate_mlp_raw = outs

        scale_msa = self._add_scalar(f"{name}/ScaleMsaPlus1", scale_msa_raw, 1.0, [1, dim])
        scale_mlp = self._add_scalar(f"{name}/ScaleMlpPlus1", scale_mlp_raw, 1.0, [1, dim])
        self.make_tanh(f"{name}/GateMsaTanh", gate_msa_raw, self.io_dtype, [1, dim])
        self.make_tanh(f"{name}/GateMlpTanh", gate_mlp_raw, self.io_dtype, [1, dim])
        gate_msa = f"{name}/GateMsaTanh/output_0"
        gate_mlp = f"{name}/GateMlpTanh/output_0"

        scale_msa = self._unsqueeze(f"{name}/ScaleMsaU", scale_msa, [1], [1, 1, dim])
        scale_mlp = self._unsqueeze(f"{name}/ScaleMlpU", scale_mlp, [1], [1, 1, dim])
        gate_msa = self._unsqueeze(f"{name}/GateMsaU", gate_msa, [1], [1, 1, dim])
        gate_mlp = self._unsqueeze(f"{name}/GateMlpU", gate_mlp, [1], [1, 1, dim])
        return scale_msa, gate_msa, scale_mlp, gate_mlp

    # ------------------------------------------------------------------
    # Attention / FeedForward / transformer block
    # ------------------------------------------------------------------
    def _rescale_preout(self, name, root_input, shape, scale=None):
        """Scale down an output-projection input by `scale` to avoid float16 overflow."""
        if self.io_dtype != ir.DataType.FLOAT16:
            return root_input
        if scale is None:
            scale = self.pre_out_proj_scale
        return self._mul(name, [root_input, self._const(self.io_dtype, scale)], shape)

    def _make_attention(self, name, x, num_tokens_shape, cos_cache, sin_cache, position_ids, attn):
        shape3 = [1, num_tokens_shape, self.dim]
        q = self._linear(f"{name}/to_q", x, attn.to_q, shape3)
        k = self._linear(f"{name}/to_k", x, attn.to_k, shape3)
        v = self._linear(f"{name}/to_v", x, attn.to_v, shape3)

        heads_shape = [1, num_tokens_shape, self.n_heads, self.head_size]
        q_heads = self._reshape(f"{name}/QHeads", q, [1, -1, self.n_heads, self.head_size], heads_shape)
        k_heads = self._reshape(f"{name}/KHeads", k, [1, -1, self.n_heads, self.head_size], heads_shape)

        if self.qk_norm:
            q_heads = self._rms_norm(f"{name}/NormQ", q_heads, attn.norm_q.weight, heads_shape)
            k_heads = self._rms_norm(f"{name}/NormK", k_heads, attn.norm_k.weight, heads_shape)

        q_flat = self._apply_rope(f"{name}/RopeQ", q_heads, num_tokens_shape, self.n_heads, cos_cache, sin_cache, position_ids)
        k_flat = self._apply_rope(f"{name}/RopeK", k_heads, num_tokens_shape, self.n_heads, cos_cache, sin_cache, position_ids)

        self.make_multi_head_attention(f"{name}/MHA", q_path=q_flat, k_path=k_flat, v_path=v)
        attn_out = f"{name}/MHA/output_0"
        self.make_value(attn_out, self.io_dtype, shape=shape3)  # re-stamp (see _linear's seq_dim note)

        attn_out = self._rescale_preout(f"{name}/to_out/PreScale", attn_out, shape3)
        return self._linear(f"{name}/to_out", attn_out, attn.to_out[0], shape3)

    def _make_feed_forward(self, name, x, num_tokens_shape, ff):
        shape3 = [1, num_tokens_shape, self.dim]
        hidden_shape = [1, num_tokens_shape, ff.w1.out_features]
        gate = self._linear(f"{name}/w1", x, ff.w1, hidden_shape)
        gate = self._silu(f"{name}/w1/Silu", gate, hidden_shape)
        up = self._linear(f"{name}/w3", x, ff.w3, hidden_shape)
        gate = self._rescale_preout(f"{name}/w1/Silu/PreScale", gate, hidden_shape, self.ff_gate_scale)
        up = self._rescale_preout(f"{name}/w3/PreScale", up, hidden_shape, self.ff_up_scale)
        gated = self._mul(f"{name}/Gated", [gate, up], hidden_shape)
        return self._linear(f"{name}/w2", gated, ff.w2, shape3)

    def _make_block(self, name, x, num_tokens_shape, cos_cache, sin_cache, position_ids, block, modulation, adaln_input=None):
        shape3 = [1, num_tokens_shape, self.dim]
        if modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self._make_adaln(
                f"{name}/adaLN_modulation", adaln_input, block.adaLN_modulation[0], self.dim
            )

        normed1 = self._rms_norm(f"{name}/attention_norm1", x, block.attention_norm1.weight, shape3)
        if modulation:
            normed1 = self._mul(f"{name}/AttnScale", [normed1, scale_msa], shape3)
        attn_out = self._make_attention(f"{name}/attention", normed1, num_tokens_shape, cos_cache, sin_cache, position_ids, block.attention)
        attn_out = self._rms_norm(f"{name}/attention_norm2", attn_out, block.attention_norm2.weight, shape3)
        if modulation:
            attn_out = self._mul(f"{name}/GateMsaMul", [attn_out, gate_msa], shape3)
        x = self._add(f"{name}/Residual1", [x, attn_out], shape3)

        normed2 = self._rms_norm(f"{name}/ffn_norm1", x, block.ffn_norm1.weight, shape3)
        if modulation:
            normed2 = self._mul(f"{name}/FfnScale", [normed2, scale_mlp], shape3)
        ffn_out = self._make_feed_forward(f"{name}/feed_forward", normed2, num_tokens_shape, block.feed_forward)
        ffn_out = self._rms_norm(f"{name}/ffn_norm2", ffn_out, block.ffn_norm2.weight, shape3)
        if modulation:
            ffn_out = self._mul(f"{name}/GateMlpMul", [ffn_out, gate_mlp], shape3)
        x = self._add(f"{name}/Residual2", [x, ffn_out], shape3)
        return x

    # ------------------------------------------------------------------
    # LayerNorm without affine params (`FinalLayer.norm_final`)
    # ------------------------------------------------------------------
    def _layer_norm_no_affine(self, name, root_input, shape, eps=1e-6):
        """`FinalLayer.norm_final` (no affine). Needs `stash_type=1` to avoid a WebGPU/WebNN
        float16 overflow."""
        dim = shape[-1]
        base = name[1:].replace("/", ".")
        scale_name, bias_name = f"{base}.scale_ones", f"{base}.bias_zeros"
        self.make_initializer(torch.ones(dim), scale_name, to=self.io_dtype)
        self.make_initializer(torch.zeros(dim), bias_name, to=self.io_dtype)
        output = f"{name}/output_0"
        self.make_node(
            "LayerNormalization",
            inputs=[root_input, scale_name, bias_name],
            outputs=[output],
            name=name,
            axis=-1,
            epsilon=eps,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape=shape)
        return output

    # ------------------------------------------------------------------
    # Sequence concatenation / slicing
    # ------------------------------------------------------------------
    def _concat_seq(self, name, parts, feat_dim, out_dim_name):
        self.make_concat(name, parts, self.io_dtype, shape=[1, out_dim_name, feat_dim], axis=1)
        return f"{name}/output_0"

    def _slice_seq(self, name, root_input, count_dim1_tensor, feat_dim, out_dim_name):
        starts = self._const(ir.DataType.INT64, [0])
        axes = self._const(ir.DataType.INT64, [1])
        self.make_slice(name, [root_input, starts, count_dim1_tensor, axes], self.io_dtype, shape=[1, out_dim_name, feat_dim])
        return f"{name}/output_0"

    # ------------------------------------------------------------------
    # Top-level graph construction
    # ------------------------------------------------------------------
    def make_model(self, input_path):
        self.make_inputs_and_outputs()
        self.weights = self.load_weights(input_path)
        self._make_rope_tables()
        self._make_timestep_freqs()

        hidden_states = self.input_names["hidden_states"]
        encoder_hidden_states = self.input_names["encoder_hidden_states"]
        timestep = self.input_names["timestep"]

        # --- dynamic dims ---
        height = self._dim_of("/model/z_image/height", hidden_states, 2, 4)
        width = self._dim_of("/model/z_image/width", hidden_states, 3, 4)
        cap_seq_len = self._dim_of("/model/z_image/cap_seq_len", encoder_hidden_states, 1, 3)

        grids = self._build_position_grids(height, width, cap_seq_len)
        h_tok, w_tok, num_img_tokens = grids["h_tok"], grids["w_tok"], grids["num_img_tokens"]

        img_freqs = self._build_freqs_cis("/model/z_image/img_freqs", *grids["img_axis"], "img_seq_len")
        cap_freqs = self._build_freqs_cis("/model/z_image/cap_freqs", *grids["cap_axis"], "cap_seq_len")
        img_cos, img_sin = self._cos_sin_caches_from_freqs_cis("/model/z_image/img_cs", img_freqs, "img_seq_len")
        cap_cos, cap_sin = self._cos_sin_caches_from_freqs_cis("/model/z_image/cap_cs", cap_freqs, "cap_seq_len")

        img_len_scalar = self._scalar("/model/z_image/img_len_scalar", num_img_tokens)
        cap_len_scalar = self._scalar("/model/z_image/cap_pos_len_scalar", cap_seq_len)
        img_pos = self._identity_position_ids("/model/z_image/img_pos", img_len_scalar, "img_seq_len")
        cap_pos = self._identity_position_ids("/model/z_image/cap_pos", cap_len_scalar, "cap_seq_len")

        # --- patchify image: [1, C, H, W] -> [1, H_t*W_t, patch_dim] ---
        img_squeezed = self._reshape(
            "/model/z_image/patchify/Squeeze", hidden_states,
            [self.in_channels, height, width], [self.in_channels, "height", "width"],
        )
        img_view = self._reshape(
            "/model/z_image/patchify/View", img_squeezed,
            [self.in_channels, h_tok, self.patch_size, w_tok, self.patch_size],
            [self.in_channels, "h_tokens", self.patch_size, "w_tokens", self.patch_size],
        )
        img_perm = "/model/z_image/patchify/Transpose"
        self.make_transpose(
            img_perm, img_view, self.io_dtype,
            ["h_tokens", "w_tokens", self.patch_size, self.patch_size, self.in_channels],
            perm=[1, 3, 2, 4, 0],
        )
        img_patches = self._reshape(
            "/model/z_image/patchify/Flatten", f"{img_perm}/output_0",
            [num_img_tokens, self.patch_dim], ["img_seq_len", self.patch_dim],
        )
        img_patches_b = self._unsqueeze("/model/z_image/patchify/Batch", img_patches, [0], [1, "img_seq_len", self.patch_dim])

        x_embedder = self.weights.all_x_embedder[self.patch_key]
        img_tokens = self._linear(
            "/model/x_embedder", img_patches_b, x_embedder, [1, "img_seq_len", self.dim],
            exclude_from_quant=True,
        )

        # --- caption embed ---
        cap_norm = self._rms_norm(
            "/model/cap_embedder.0", encoder_hidden_states, self.weights.cap_embedder[0].weight,
            [1, "cap_seq_len", self.cap_feat_dim],
        )
        cap_tokens = self._linear(
            "/model/cap_embedder.1", cap_norm, self.weights.cap_embedder[1], [1, "cap_seq_len", self.dim],
            exclude_from_quant=True,
        )

        # --- timestep / AdaLN input ---
        adaln_input = self._make_timestep_embedding(timestep, self.weights.t_embedder)

        # --- noise refiner (image tokens only) ---
        x = img_tokens
        for i in range(self.n_refiner_layers):
            x = self._make_block(
                f"/model/noise_refiner.{i}", x, "img_seq_len", img_cos, img_sin, img_pos,
                self.weights.noise_refiner[i], modulation=True, adaln_input=adaln_input,
            )
        img_tokens = x

        # --- context refiner (caption tokens only) ---
        x = cap_tokens
        for i in range(self.n_refiner_layers):
            x = self._make_block(
                f"/model/context_refiner.{i}", x, "cap_seq_len", cap_cos, cap_sin, cap_pos,
                self.weights.context_refiner[i], modulation=False,
            )
        cap_tokens = x

        # --- unify: image tokens first, caption tokens second ---
        unified = self._concat_seq("/model/z_image/unify_tokens", [img_tokens, cap_tokens], self.dim, "unified_seq_len")
        half = self.head_size // 2
        self.make_concat("/model/z_image/unify_cos", [img_cos, cap_cos], self.io_dtype, shape=["unified_seq_len", half], axis=0)
        self.make_concat("/model/z_image/unify_sin", [img_sin, cap_sin], self.io_dtype, shape=["unified_seq_len", half], axis=0)
        unified_cos = "/model/z_image/unify_cos/output_0"
        unified_sin = "/model/z_image/unify_sin/output_0"
        unified_len_scalar = self._int_scalar_add("/model/z_image/unified_len_scalar", img_len_scalar, cap_len_scalar)
        unified_pos = self._identity_position_ids("/model/z_image/unified_pos", unified_len_scalar, "unified_seq_len")

        # --- main layers (unified sequence) ---
        x = unified
        for i in range(self.n_layers):
            x = self._make_block(
                f"/model/layers.{i}", x, "unified_seq_len", unified_cos, unified_sin, unified_pos,
                self.weights.layers[i], modulation=True, adaln_input=adaln_input,
            )

        # --- final layer (AdaLN + LayerNorm(no affine) + Linear) ---
        final_layer = self.weights.all_final_layer[self.patch_key]
        final_silu = self._silu("/model/final_layer/adaLN_modulation/Silu", adaln_input, [1, self.adaln_embed_dim])
        final_scale_raw = self._linear(
            "/model/final_layer/adaLN_modulation.1", final_silu, final_layer.adaLN_modulation[1], [1, self.dim],
            exclude_from_quant=True,
        )
        final_scale = self._add_scalar("/model/final_layer/adaLN_modulation.1/Plus1", final_scale_raw, 1.0, [1, self.dim])
        final_scale = self._unsqueeze("/model/final_layer/adaLN_modulation.1/Unsqueeze", final_scale, [1], [1, 1, self.dim])

        normed = self._layer_norm_no_affine("/model/final_layer/norm_final", x, [1, "unified_seq_len", self.dim])
        scaled = self._mul("/model/final_layer/ScaleMul", [normed, final_scale], [1, "unified_seq_len", self.dim])
        out_patches = self._linear(
            "/model/final_layer/linear", scaled, final_layer.linear, [1, "unified_seq_len", self.patch_dim]
        )

        # --- unpatchify: image tokens are the first `num_img_tokens` rows ---
        img_out = self._slice_seq("/model/z_image/unpatchify/Slice", out_patches, num_img_tokens, self.patch_dim, "img_seq_len")
        img_out = self._reshape(
            "/model/z_image/unpatchify/Squeeze", img_out, [num_img_tokens, self.patch_dim], ["img_seq_len", self.patch_dim]
        )
        img_view_out = self._reshape(
            "/model/z_image/unpatchify/View", img_out,
            [h_tok, w_tok, self.patch_size, self.patch_size, self.in_channels],
            ["h_tokens", "w_tokens", self.patch_size, self.patch_size, self.in_channels],
        )
        img_perm_out = "/model/z_image/unpatchify/Transpose"
        self.make_transpose(
            img_perm_out, img_view_out, self.io_dtype,
            [self.in_channels, "h_tokens", self.patch_size, "w_tokens", self.patch_size],
            perm=[4, 0, 2, 1, 3],
        )
        img_merged = self._reshape(
            "/model/z_image/unpatchify/Merge", f"{img_perm_out}/output_0",
            [self.in_channels, height, width], [self.in_channels, "height", "width"],
        )
        sample = self._unsqueeze("/model/z_image/unpatchify/Batch", img_merged, [0], [1, self.in_channels, "height", "width"])

        self.make_node("Identity", inputs=[sample], outputs=["sample"], name="/model/z_image/output_identity")

        del self.weights


# Standalone build driver (replaces builder.py's create_model, scoped to this one architecture).
def build(input_path, output_dir, precision="f16_int4_quant", extra_options=None):
    if precision not in PRECISION_CONFIGS:
        raise ValueError(f"Unknown precision '{precision}'; choose from {sorted(PRECISION_CONFIGS)}")

    precision_config = PRECISION_CONFIGS[precision]
    extra_options = dict(extra_options or {})
    extra_options.setdefault("hf_remote", False)
    if precision_config["int4_quant"]:
        extra_options.setdefault("block_size", 32)
        extra_options.setdefault("accuracy_level", 4)
        # MatMul only -- Gather would int4-quantize the RoPE frequency tables (corrupting them).
        extra_options.setdefault("op_types_to_quantize", ("MatMul",))
        if precision == "f32_int4_quant":
            extra_options.setdefault("use_webgpu_fp32", True)

    extra_options["filename"] = f"transformer_model_{FILENAME_SUFFIXES[precision]}.onnx"

    builder_precision = precision_config["builder_precision"]
    io_dtype = set_io_dtype(builder_precision, "webgpu", extra_options)
    onnx_dtype = set_onnx_dtype(builder_precision, extra_options)

    config = load_diffusers_config(input_path)

    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    model = ZImageTransformerModel(config, io_dtype, onnx_dtype, "webgpu", cache_dir, extra_options)
    model.make_model(input_path)
    model.save_model(onnx_dir)
    return onnx_dir


def get_args():
    parser = argparse.ArgumentParser(description="Export the Z-Image-Turbo transformer trunk to ONNX.")
    parser.add_argument("input_path", help="Path to the Z-Image-Turbo `transformer/` folder (or its parent).")
    parser.add_argument(
        "-o", "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the ONNX model to (under an `onnx/` subdir). Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-p", "--precision", default="f16_int4_quant", choices=sorted(PRECISION_CONFIGS),
        help="Output precision/quantization. Default: f16_int4_quant.",
    )
    parser.add_argument(
        "--extra_options", nargs="*", default=[],
        help="Extra key=value options, e.g. block_size=32 accuracy_level=4.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    input_path = args.input_path
    if os.path.isdir(os.path.join(input_path, "transformer")):
        input_path = os.path.join(input_path, "transformer")
    onnx_dir = build(input_path, args.output_dir, args.precision, parse_extra_options(args.extra_options))
    print(f"\nSuccess: transformer exported to {onnx_dir}")


if __name__ == "__main__":
    main()
