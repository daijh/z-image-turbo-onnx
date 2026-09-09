# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Intel Corporation. All rights reserved.
# --------------------------------------------------------------------------
"""Standalone exporter for the Z-Image-Turbo VAE decoder (`AutoencoderKL.decoder`)."""

import argparse
import json
import os
import types
from typing import Any

import onnx_ir as ir
import torch
from onnxruntime_genai.models.builders.base import Model

PRECISION_CONFIGS = {
    "f16": {"builder_precision": "fp16"},
    "f32": {"builder_precision": "fp32"},
}

DEFAULT_OUTPUT_DIR = "z-image-turbo-onnx"


# ---------------------------------------------------------------------------
# Vendored from onnxruntime_genai.models.builder.
# ---------------------------------------------------------------------------
def set_io_dtype(precision, execution_provider, extra_options) -> ir.DataType:
    cpu_quant = precision in {"int4", "int8"} and execution_provider == "cpu"
    fp32_webgpu = execution_provider == "webgpu" and extra_options.get("use_webgpu_fp32", False)
    bf16_cuda = precision == "int4" and execution_provider in {"cuda", "trt-rtx"} and extra_options.get("use_cuda_bf16", False)

    if precision == "fp32" or cpu_quant or fp32_webgpu:
        return ir.DataType.FLOAT
    if precision == "bf16" or bf16_cuda:
        return ir.DataType.BFLOAT16
    return ir.DataType.FLOAT16


def set_onnx_dtype(precision: str, extra_options: dict[str, Any]) -> ir.DataType:
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
    config_path = os.path.join(input_path, "config.json")
    with open(config_path) as f:
        raw_config = json.load(f)
    raw_config.setdefault("_name_or_path", input_path)
    return types.SimpleNamespace(**raw_config)


def parse_extra_options(pairs):
    extra_options = {}
    for kv_str in pairs or []:
        key, _, value = kv_str.partition("=")
        extra_options[key.strip()] = value.strip()

    for bool_key in ("hf_remote", "use_webgpu_fp32", "fuse_group_norm"):
        if bool_key in extra_options:
            extra_options[bool_key] = extra_options[bool_key] not in ("false", "False", "0")

    return extra_options


class ZImageVAEDecoderModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        mid_channels = int(config.block_out_channels[-1])
        fake_config = types.SimpleNamespace(
            _name_or_path=getattr(config, "_name_or_path", "z-image-vae-decoder"),
            architectures=["AutoencoderKL"],
            hidden_size=mid_channels,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_hidden_layers=1,
            intermediate_size=mid_channels,
            vocab_size=0,
            hidden_act="silu",
            max_position_embeddings=1,
            rms_norm_eps=1e-6,
        )
        super().__init__(fake_config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.model_type = "z-image-vae-decoder"
        # Spatial self-attention: bidirectional, no KV cache.
        self.attention_attrs["unidirectional"] = False

        self.mid_channels = mid_channels
        self.latent_channels = int(config.latent_channels)
        self.out_channels = int(config.out_channels)
        self.block_out_channels = [int(c) for c in config.block_out_channels]
        self.norm_num_groups = int(config.norm_num_groups)
        # False (default): decomposed standard-ONNX GroupNorm; True: com.microsoft GroupNorm/SkipGroupNorm.
        self.fuse_group_norm = bool(extra_options.get("fuse_group_norm", False))
        self._io_dtype = self.io_dtype

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(self, input_path):
        from diffusers import AutoencoderKL

        # Load from `input_path` (the vae/ dir) rather than `self.model_name_or_path`.
        model = AutoencoderKL.from_pretrained(
            input_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
        )
        model.eval()
        return model.decoder

    # ------------------------------------------------------------------
    # Inputs / outputs
    # ------------------------------------------------------------------
    def make_inputs_and_outputs(self):
        self.input_names = {"latent_sample": "latent_sample"}
        self.output_names = {"sample": "sample"}

        latent = self.make_value(
            "latent_sample", self._io_dtype,
            shape=[1, self.latent_channels, "latent_height", "latent_width"],
        )
        self.model.graph.inputs.extend([latent])

        sample = self.make_value(
            "sample", self._io_dtype, shape=[1, self.out_channels, "height", "width"]
        )
        self.model.graph.outputs.append(sample)

    # ------------------------------------------------------------------
    # Small node-building helpers
    # ------------------------------------------------------------------
    def _init_name(self, node_name, suffix):
        return node_name[1:].replace("/", ".") + suffix

    def _const(self, dtype, value):
        return f"/model/constants/{self.to_str_dtype(dtype)}/{value}"

    def _transpose(self, name, root_input, perm):
        self.make_transpose(name, root_input, self._io_dtype, shape=None, perm=perm)
        return f"{name}/output_0"

    def _add(self, name, inputs):
        self.make_add(name, inputs, self._io_dtype, shape=None)
        return f"{name}/output_0"

    def _mul(self, name, inputs):
        self.make_mul(name, inputs, self._io_dtype, shape=None)
        return f"{name}/output_0"

    def _reshape(self, name, root_input, shape_tensor_name):
        self.make_reshape(name, [root_input, shape_tensor_name], self._io_dtype, shape=None)
        return f"{name}/output_0"

    def _conv(self, name, root_input, conv):
        weight = self._init_name(name, ".weight")
        self.make_initializer(conv.weight, weight, to=self._io_dtype)
        inputs = [root_input, weight]
        if conv.bias is not None:
            bias = self._init_name(name, ".bias")
            self.make_initializer(conv.bias, bias, to=self._io_dtype)
            inputs.append(bias)

        kh, kw = conv.kernel_size
        ph, pw = conv.padding
        sh, sw = conv.stride
        dh, dw = conv.dilation
        self.make_conv(
            name, inputs, self._io_dtype, shape=None,
            strides=[sh, sw], pads=[ph, pw, ph, pw], dilations=[dh, dw],
            group=conv.groups, kernel_shape=[kh, kw],
        )
        return f"{name}/output_0"

    def _linear(self, name, root_input, linear):
        """Attention Q/K/V/out projection. Goes through `make_matmul` so `-p int4/int8` applies."""
        self.make_matmul(linear, name, root_input, seq_dim="attn_seq")
        output = f"{name}/output_0"
        if linear.bias is not None:
            self.make_add_bias(linear.bias, f"{name}/Add", output, seq_dim="attn_seq")
            output = f"{name}/Add/output_0"
        return output

    # ------------------------------------------------------------------
    # GroupNorm (fused contrib op or decomposed standard ops)
    # ------------------------------------------------------------------
    def _group_norm(self, name, source, gnmod, swish, expose_skip=True):
        if self.fuse_group_norm:
            return self._group_norm_fused(name, source, gnmod, swish, expose_skip)

        skip_sum = None
        if isinstance(source, tuple):
            conv_branch, skip_branch = source
            skip_sum = self._add(f"{name}/skip_add", [conv_branch, skip_branch])
            x = skip_sum
        else:
            x = source
        normed = self._group_norm_decomposed(name, x, gnmod, swish)
        return normed, (skip_sum if expose_skip else None)

    def _group_norm_decomposed(self, name, x_nchw, gnmod, swish):
        groups = int(gnmod.num_groups)
        channels = int(gnmod.num_channels)
        eps = float(gnmod.eps)

        gamma = self._init_name(name, ".weight")
        beta = self._init_name(name, ".bias")
        self.make_initializer(gnmod.weight.detach().reshape(channels, 1, 1), gamma, to=self._io_dtype)
        self.make_initializer(gnmod.bias.detach().reshape(channels, 1, 1), beta, to=self._io_dtype)
        in_scale = f"{name}/instnorm_scale"
        in_bias = f"{name}/instnorm_bias"
        self.make_initializer(torch.ones(groups), in_scale, to=self._io_dtype)
        self.make_initializer(torch.zeros(groups), in_bias, to=self._io_dtype)

        self.make_shape(f"{name}/in_shape", x_nchw, shape=[4])
        grouped = self._reshape(
            f"{name}/to_groups", x_nchw, self._const(ir.DataType.INT64, [0, groups, -1])
        )
        in_normed = f"{name}/InstanceNormalization/output_0"
        self.make_node(
            "InstanceNormalization", inputs=[grouped, in_scale, in_bias], outputs=[in_normed],
            name=f"{name}/InstanceNormalization", epsilon=eps,
        )
        self.make_value(in_normed, self._io_dtype)
        normed = self._reshape(f"{name}/from_groups", in_normed, f"{name}/in_shape/output_0")
        normed = self._mul(f"{name}/Mul", [normed, gamma])
        normed = self._add(f"{name}/Add", [normed, beta])
        if swish:
            self.make_sigmoid(f"{name}/Sigmoid", normed, self._io_dtype, shape=None)
            normed = self._mul(f"{name}/silu", [normed, f"{name}/Sigmoid/output_0"])
        return normed

    def _group_norm_fused(self, name, source, gnmod, swish, expose_skip):
        gamma = self._init_name(name, ".weight")
        beta = self._init_name(name, ".bias")
        self.make_initializer(gnmod.weight, gamma, to=self._io_dtype)
        self.make_initializer(gnmod.bias, beta, to=self._io_dtype)
        groups = int(gnmod.num_groups)
        eps = float(gnmod.eps)
        activation = 1 if swish else 0

        if isinstance(source, tuple):
            conv_branch, skip_branch = source
            x_nhwc = self._transpose(f"{name}/x_t", conv_branch, [0, 2, 3, 1])
            skip_nhwc = self._transpose(f"{name}/skip_t", skip_branch, [0, 2, 3, 1])
            normed_nhwc = f"{name}/output_0"
            outputs = [normed_nhwc]
            skip_sum_nhwc = None
            if expose_skip:
                skip_sum_nhwc = f"{name}/sum"
                outputs.append(skip_sum_nhwc)
            self.make_node(
                "SkipGroupNorm", inputs=[x_nhwc, gamma, beta, skip_nhwc], outputs=outputs,
                name=name, domain="com.microsoft",
                activation=activation, channels_last=1, epsilon=eps, groups=groups,
            )
            for output in outputs:
                self.make_value(output, self._io_dtype)
            normed = self._transpose(f"{name}/post_t", normed_nhwc, [0, 3, 1, 2])
            skip_sum = self._transpose(f"{name}/sum_post_t", skip_sum_nhwc, [0, 3, 1, 2]) if skip_sum_nhwc else None
            return normed, skip_sum

        x_nhwc = self._transpose(f"{name}/x_t", source, [0, 2, 3, 1])
        normed_nhwc = f"{name}/output_0"
        self.make_node(
            "GroupNorm", inputs=[x_nhwc, gamma, beta], outputs=[normed_nhwc],
            name=name, domain="com.microsoft",
            activation=activation, channels_last=1, epsilon=eps, groups=groups,
        )
        self.make_value(normed_nhwc, self._io_dtype)
        normed = self._transpose(f"{name}/post_t", normed_nhwc, [0, 3, 1, 2])
        return normed, None

    # ------------------------------------------------------------------
    # ResNet block
    # ------------------------------------------------------------------
    def _resnet(self, name, source, resnet):
        normed, skip_sum = self._group_norm(f"{name}/norm1", source, resnet.norm1, swish=True, expose_skip=True)
        if skip_sum is not None:
            # SkipGroupNorm case: identity shortcut, residual base is the fused skip sum.
            residual_base = skip_sum
        elif resnet.conv_shortcut is not None:
            residual_base = self._conv(f"{name}/conv_shortcut", source, resnet.conv_shortcut)
        else:
            residual_base = source

        hidden = self._conv(f"{name}/conv1", normed, resnet.conv1)
        normed2, _ = self._group_norm(f"{name}/norm2", hidden, resnet.norm2, swish=True)
        hidden = self._conv(f"{name}/conv2", normed2, resnet.conv2)
        return (hidden, residual_base)

    # ------------------------------------------------------------------
    # Mid-block self-attention (unfused, single-head)
    # ------------------------------------------------------------------
    def _mid_attention(self, name, x_nchw, attn):
        assert attn.heads == 1, f"expected single-head VAE attention, got heads={attn.heads}"
        normed, _ = self._group_norm(f"{name}/group_norm", x_nchw, attn.group_norm, swish=False)

        to_seq_shape = self._const(ir.DataType.INT64, [0, self.mid_channels, -1])
        flat = self._reshape(f"{name}/flatten", normed, to_seq_shape)
        seq = self._transpose(f"{name}/to_seq", flat, [0, 2, 1])

        q = self._linear(f"{name}/to_q", seq, attn.to_q)
        k = self._linear(f"{name}/to_k", seq, attn.to_k)
        v = self._linear(f"{name}/to_v", seq, attn.to_v)

        scale = self._const(self._io_dtype, self.mid_channels ** -0.25)
        q = self._mul(f"{name}/q_scale", [q, scale])
        k_t = self._transpose(f"{name}/kT", k, [0, 2, 1])  # [1, C, H*W]
        k_t = self._mul(f"{name}/k_scale", [k_t, scale])
        scores = f"{name}/scores/output_0"
        self.make_node("MatMul", inputs=[q, k_t], outputs=[scores], name=f"{name}/scores")
        self.make_value(scores, self._io_dtype)
        probs = f"{name}/softmax/output_0"
        self.make_node("Softmax", inputs=[scores], outputs=[probs], name=f"{name}/softmax", axis=-1)
        self.make_value(probs, self._io_dtype)
        ctx = f"{name}/context/output_0"
        self.make_node("MatMul", inputs=[probs, v], outputs=[ctx], name=f"{name}/context")
        self.make_value(ctx, self._io_dtype)
        attn_out = self._linear(f"{name}/to_out.0", ctx, attn.to_out[0])

        back_seq = self._transpose(f"{name}/from_seq", attn_out, [0, 2, 1])
        self.make_shape(f"{name}/in_shape", x_nchw, shape=[4])
        spatial = self._reshape(f"{name}/unflatten", back_seq, f"{name}/in_shape/output_0")

        return self._add(f"{name}/Add", [spatial, x_nchw])

    # ------------------------------------------------------------------
    # Upsampler (nearest 2x, then conv)
    # ------------------------------------------------------------------
    def _upsample(self, name, x_nchw, conv):
        if self.fuse_group_norm:
            x = self._transpose(f"{name}/pre_t", x_nchw, [0, 2, 3, 1])
            scales = self._const(ir.DataType.FLOAT, [1.0, 2.0, 2.0, 1.0])
        else:
            x = x_nchw
            scales = self._const(ir.DataType.FLOAT, [1.0, 1.0, 2.0, 2.0])
        resize_out = f"{name}/Resize/output_0"
        self.make_node(
            "Resize", inputs=[x, "", scales], outputs=[resize_out], name=f"{name}/Resize",
            coordinate_transformation_mode="asymmetric", cubic_coeff_a=-0.75,
            mode="nearest", nearest_mode="floor",
        )
        self.make_value(resize_out, self._io_dtype)
        y_nchw = self._transpose(f"{name}/post_t", resize_out, [0, 3, 1, 2]) if self.fuse_group_norm else resize_out
        return self._conv(f"{name}/conv", y_nchw, conv)

    # ------------------------------------------------------------------
    # Top-level graph construction
    # ------------------------------------------------------------------
    def make_model(self, input_path):
        self.make_inputs_and_outputs()
        self.weights = self.load_weights(input_path)
        dec = self.weights

        x = self.input_names["latent_sample"]

        # --- conv_in ---
        x = self._conv("/decoder/conv_in", x, dec.conv_in)

        # --- mid block: resnet0 -> attention -> resnet1 ---
        mid = dec.mid_block
        r0 = self._resnet("/decoder/mid_block/resnets.0", x, mid.resnets[0])
        x = self._add("/decoder/mid_block/resnets.0/Add", list(r0))  # feeds attention (non-norm)
        x = self._mid_attention("/decoder/mid_block/attentions.0", x, mid.attentions[0])
        source = self._resnet("/decoder/mid_block/resnets.1", x, mid.resnets[1])

        # --- up blocks ---
        for bi, up in enumerate(dec.up_blocks):
            n_res = len(up.resnets)
            for rj in range(n_res):
                r = self._resnet(f"/decoder/up_blocks.{bi}/resnets.{rj}", source, up.resnets[rj])
                if rj < n_res - 1:
                    source = r
                elif getattr(up, "upsamplers", None):
                    x = self._add(f"/decoder/up_blocks.{bi}/resnets.{rj}/Add", list(r))
                    x = self._upsample(f"/decoder/up_blocks.{bi}/upsamplers.0", x, up.upsamplers[0].conv)
                    source = x
                else:
                    source = r

        # --- conv_norm_out (SkipGroupNorm, swish) -> conv_out ---
        y, _ = self._group_norm(
            "/decoder/conv_norm_out", source, dec.conv_norm_out, swish=True, expose_skip=False
        )
        out = self._conv("/decoder/conv_out", y, dec.conv_out)

        self.make_node("Identity", inputs=[out], outputs=["sample"], name="/decoder/output_identity")

        del self.weights

    # ------------------------------------------------------------------
    # Save (single self-contained .onnx file)
    # ------------------------------------------------------------------
    def save_model(self, out_dir):
        print(f"Saving ONNX model in {out_dir}")
        already_quantized_in_qdq_format = self.quant_type is not None and self.quant_attrs["use_qdq"]
        if self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4, ir.DataType.INT8, ir.DataType.UINT8} and not already_quantized_in_qdq_format:
            model = self.to_nbits()
        else:
            model = self.model

        model.graph.sort()

        out_path = os.path.join(out_dir, self.filename)
        data_path = out_path + ".data"
        for stale in (out_path, data_path):
            if os.path.exists(stale):
                print(f"Overwriting {stale}")
                os.remove(stale)

        ir.save(model, out_path)

        if os.path.isdir(self.cache_dir) and not os.listdir(self.cache_dir):
            os.rmdir(self.cache_dir)


# ---------------------------------------------------------------------------
# Standalone build driver (replaces builder.py's `create_model`, scoped to
# just this one architecture)
# ---------------------------------------------------------------------------
def build(input_path, output_dir, precision="f16", extra_options=None):
    if precision not in PRECISION_CONFIGS:
        raise ValueError(f"Unknown precision '{precision}'; choose from {sorted(PRECISION_CONFIGS)}")

    precision_config = PRECISION_CONFIGS[precision]
    extra_options = dict(extra_options or {})
    extra_options.setdefault("hf_remote", False)
    extra_options["filename"] = f"vae_decoder_model_{precision}.onnx"

    builder_precision = precision_config["builder_precision"]
    io_dtype = set_io_dtype(builder_precision, "webgpu", extra_options)
    onnx_dtype = set_onnx_dtype(builder_precision, extra_options)

    config = load_diffusers_config(input_path)

    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    model = ZImageVAEDecoderModel(config, io_dtype, onnx_dtype, "webgpu", cache_dir, extra_options)
    model.make_model(input_path)
    model.save_model(onnx_dir)
    return onnx_dir


def get_args():
    parser = argparse.ArgumentParser(description="Export the Z-Image-Turbo VAE decoder to ONNX.")
    parser.add_argument("input_path", help="Path to the Z-Image-Turbo `vae/` folder (or its parent).")
    parser.add_argument(
        "-o", "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the ONNX model to (under an `onnx/` subdir). Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-p", "--precision", default="f16", choices=sorted(PRECISION_CONFIGS),
        help="Output precision. Default: f16.",
    )
    parser.add_argument(
        "--extra_options", nargs="*", default=[],
        help="Extra key=value options, e.g. fuse_group_norm=true.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    input_path = args.input_path
    if os.path.isdir(os.path.join(input_path, "vae")):
        input_path = os.path.join(input_path, "vae")
    onnx_dir = build(input_path, args.output_dir, args.precision, parse_extra_options(args.extra_options))
    print(f"\nSuccess: vae_decoder exported to {onnx_dir}")


if __name__ == "__main__":
    main()
