# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Intel Corporation. All rights reserved.
# --------------------------------------------------------------------------
"""Standalone exporter for the Z-Image-Turbo NSFW safety checker."""

import argparse
import os

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import torch
import torch.nn as nn
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
    cosine_distance,
)

from build_helper_models import convert_to_f16
from external_data_utils import (
    INLINE_SIZE_THRESHOLD_BYTES,
    MAX_SHARD_SIZE_BYTES,
    save_ir_model_sharded,
)

DEFAULT_OUTPUT_DIR = "z-image-turbo-onnx"

PRECISION_CONFIGS = {"f16": {}, "f32": {}}


def parse_extra_options(pairs):
    """Parse `key=value` strings (as passed via `--extra_options`) into a dict."""
    extra_options = {}
    for kv_str in pairs or []:
        key, _, value = kv_str.partition("=")
        extra_options[key.strip()] = value.strip()
    return extra_options


def optimize_onnx_graph(onnx_path):
    """Run ORT's EP-neutral (BASIC-level) graph optimizations on `onnx_path`, in place."""
    tmp_path = onnx_path + ".opt.tmp"
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = tmp_path
    ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    os.replace(tmp_path, onnx_path)
    print(f"  optimized {onnx_path} ({os.path.getsize(onnx_path) / (1024 * 1024):.1f} MB)")


class SafetyCheckerOnnxWrapper(nn.Module):
    """`StableDiffusionSafetyChecker.forward_onnx` minus the `images` masking I/O."""

    def __init__(self, safety_checker):
        super().__init__()
        self.vision_model = safety_checker.vision_model
        self.visual_projection = safety_checker.visual_projection
        self.concept_embeds = safety_checker.concept_embeds
        self.special_care_embeds = safety_checker.special_care_embeds
        self.concept_embeds_weights = safety_checker.concept_embeds_weights
        self.special_care_embeds_weights = safety_checker.special_care_embeds_weights

    def forward(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        special_scores = special_cos_dist - self.special_care_embeds_weights
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        return has_nsfw_concepts


def build(input_path, output_dir, precision="f16", extra_options=None, opset=17):
    if precision not in PRECISION_CONFIGS:
        raise ValueError(f"Unknown precision '{precision}'; choose from {sorted(PRECISION_CONFIGS)}")

    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    print(f"Loading safety checker checkpoint from {input_path}")
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(input_path)
    safety_checker.eval()
    wrapper = SafetyCheckerOnnxWrapper(safety_checker).eval()

    f32_path = os.path.join(onnx_dir, "safety_checker_model_f32.onnx")
    dummy_clip_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        wrapper,
        (dummy_clip_input,),
        f32_path,
        input_names=["clip_input"],
        output_names=["has_nsfw_concepts"],
        dynamic_axes={},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  wrote {f32_path} ({os.path.getsize(f32_path) / (1024 * 1024):.1f} MB)")

    optimize_onnx_graph(f32_path)

    if precision == "f32":
        final_name = "safety_checker_model_f32.onnx"
        proto = onnx.load(f32_path)
    else:
        final_name = "safety_checker_model_f16.onnx"
        f16_tmp = os.path.join(onnx_dir, "safety_checker_model_f16_tmp.onnx")
        convert_to_f16(f32_path, f16_tmp)
        proto = onnx.load(f16_tmp)
        os.remove(f16_tmp)
    os.remove(f32_path)

    save_ir_model_sharded(
        ir.from_proto(proto), onnx_dir, final_name,
        size_threshold_bytes=INLINE_SIZE_THRESHOLD_BYTES,
        max_shard_size_bytes=MAX_SHARD_SIZE_BYTES,
    )
    final_path = os.path.join(onnx_dir, final_name)
    print(f"  wrote {final_path} (+ sharded external data)")

    sess = ort.InferenceSession(final_path, providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"input: {i}")
    for o in sess.get_outputs():
        print(f"output: {o}")

    dtype = np.float16 if precision == "f16" else np.float32
    sample = np.random.randn(1, 3, 224, 224).astype(dtype)
    outputs = sess.run(None, {"clip_input": sample})
    print(
        f"  sanity run output 'has_nsfw_concepts': shape={outputs[0].shape}, "
        f"dtype={outputs[0].dtype}, values={outputs[0]}"
    )

    return onnx_dir


def get_args():
    parser = argparse.ArgumentParser(description="Export the Z-Image-Turbo NSFW safety checker to ONNX.")
    parser.add_argument(
        "input_path",
        help="Path to a local CompVis/stable-diffusion-safety-checker-compatible checkpoint "
        "(config.json + weights), e.g. via huggingface_hub.snapshot_download('CompVis/"
        "stable-diffusion-safety-checker', local_dir=...).",
    )
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
        help="Extra key=value options (currently unused; kept for CLI consistency with the "
        "other build_*.py scripts).",
    )
    return parser.parse_args()


def main():
    args = get_args()
    onnx_dir = build(args.input_path, args.output_dir, args.precision, parse_extra_options(args.extra_options))
    print(f"\nSuccess: safety_checker exported to {onnx_dir}")


if __name__ == "__main__":
    main()
