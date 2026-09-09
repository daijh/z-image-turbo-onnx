# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Intel Corporation. All rights reserved.
# --------------------------------------------------------------------------
"""Main entry point for exporting the Z-Image-Turbo pipeline to standalone ONNX models."""

import argparse
import os
import shutil
import subprocess
import sys

import build_helper_models
import build_safety_checker
import build_text_encoder
import build_transformer
import build_vae_decoder

ALL_COMPONENTS = ("transformer", "text_encoder", "vae_decoder", "helper_models", "safety_checker")

# Small tokenizer files that AutoTokenizer.from_pretrained needs; they live in the checkpoint's
# sibling `tokenizer/` folder, not `text_encoder/`.
TOKENIZER_FILES = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def get_args():
    parser = argparse.ArgumentParser(description="Export the Z-Image-Turbo pipeline to ONNX.")
    parser.add_argument("input_path", help="Path to the Z-Image-Turbo checkpoint (repo root or component subfolder).")
    parser.add_argument(
        "-o", "--output_dir", default=build_transformer.DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the ONNX model(s) to (under an `onnx/` subdir). Default: {build_transformer.DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-m", "--model", default="all",
        choices=["transformer", "vae_decoder", "helper_models", "text_encoder", "safety_checker", "all"],
        help="Which component to build. Default: all.",
    )
    parser.add_argument(
        "--extra_options", nargs="*", default=[],
        help="Extra key=value options passed through to the component's builder.",
    )
    parser.add_argument(
        "--safety_checker_checkpoint", default="",
        help="Path to a local CompVis/stable-diffusion-safety-checker-compatible checkpoint "
        "(e.g. via huggingface_hub.snapshot_download('CompVis/stable-diffusion-safety-checker', "
        "local_dir=...)). This is a separate download from input_path. Required for "
        "-m safety_checker; for -m all, safety_checker is skipped if this isn't given.",
    )
    return parser.parse_args()


def build_one(model, input_path, output_dir, extra_options, safety_checker_checkpoint=""):
    if model == "transformer":
        transformer_input = input_path
        if os.path.isdir(os.path.join(input_path, "transformer")):
            transformer_input = os.path.join(input_path, "transformer")
        return build_transformer.build(
            transformer_input, output_dir, extra_options=build_transformer.parse_extra_options(extra_options)
        )
    if model == "text_encoder":
        text_encoder_input = input_path
        if os.path.isdir(os.path.join(input_path, "text_encoder")):
            text_encoder_input = os.path.join(input_path, "text_encoder")
        return build_text_encoder.build(
            text_encoder_input, output_dir, extra_options=build_text_encoder.parse_extra_options(extra_options)
        )
    if model == "vae_decoder":
        vae_input = input_path
        if os.path.isdir(os.path.join(input_path, "vae")):
            vae_input = os.path.join(input_path, "vae")
        return build_vae_decoder.build(
            vae_input, output_dir, extra_options=build_vae_decoder.parse_extra_options(extra_options)
        )
    if model == "helper_models":
        return build_helper_models.build(
            input_path, output_dir, extra_options=build_helper_models.parse_extra_options(extra_options)
        )
    if model == "safety_checker":
        if not safety_checker_checkpoint:
            raise NotImplementedError(
                "-m safety_checker requires --safety_checker_checkpoint (a separate local "
                "CompVis/stable-diffusion-safety-checker-compatible checkpoint, not input_path)."
            )
        return build_safety_checker.build(
            safety_checker_checkpoint, output_dir, extra_options=build_safety_checker.parse_extra_options(extra_options)
        )
    raise ValueError(f"Unknown -m/--model value: {model}")


def resolve_tokenizer_dir(input_path):
    text_encoder_dir = input_path
    if os.path.isdir(os.path.join(input_path, "text_encoder")):
        text_encoder_dir = os.path.join(input_path, "text_encoder")
    return os.path.join(os.path.dirname(os.path.normpath(text_encoder_dir)), "tokenizer")


def _link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_tokenizer(input_path, output_dir):
    tokenizer_dir = resolve_tokenizer_dir(input_path)
    if not os.path.isdir(tokenizer_dir):
        print(f"Skipping tokenizer copy: {tokenizer_dir} not found", file=sys.stderr)
        return

    dest_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(dest_dir, exist_ok=True)
    missing = []
    for fname in TOKENIZER_FILES:
        src = os.path.join(tokenizer_dir, fname)
        if os.path.isfile(src):
            _link_or_copy(src, os.path.join(dest_dir, fname))
        else:
            missing.append(fname)

    if missing:
        print(f"Warning: tokenizer files not found in {tokenizer_dir}: {missing}", file=sys.stderr)
    else:
        print(f"Copied tokenizer to {dest_dir}")


def build_component_subprocess(model, args):
    cmd = [sys.executable, os.path.abspath(__file__), args.input_path, "-m", model, "-o", args.output_dir]
    if args.extra_options:
        cmd += ["--extra_options", *args.extra_options]
    if args.safety_checker_checkpoint:
        cmd += ["--safety_checker_checkpoint", args.safety_checker_checkpoint]
    return subprocess.run(cmd).returncode


def build_all(args):
    results = {}
    for model in ALL_COMPONENTS:
        if model == "safety_checker" and not args.safety_checker_checkpoint:
            print(f"\nSkipping {model}: no --safety_checker_checkpoint given")
            results[model] = "skipped"
            continue
        print(f"\n=== Building {model} (isolated subprocess) ===")
        returncode = build_component_subprocess(model, args)
        results[model] = "ok" if returncode == 0 else f"FAILED (exit {returncode})"

    print("\n=== Summary ===")
    for model in ALL_COMPONENTS:
        print(f"  {model:<14} {results.get(model, 'skipped')}")

    failed = [m for m, status in results.items() if status.startswith("FAILED")]
    return results, failed


def main():
    args = get_args()
    if args.model == "all":
        _results, failed = build_all(args)
        copy_tokenizer(args.input_path, args.output_dir)
        onnx_dir = os.path.join(args.output_dir, "onnx")
        if failed:
            print(f"\nFailure: {len(failed)} component(s) failed: {', '.join(failed)}", file=sys.stderr)
            sys.exit(1)
        print(f"\nSuccess: all exported to {onnx_dir}")
    else:
        onnx_dir = build_one(
            args.model, args.input_path, args.output_dir, args.extra_options, args.safety_checker_checkpoint
        )
        copy_tokenizer(args.input_path, args.output_dir)
        print(f"\nSuccess: {args.model} exported to {onnx_dir}")


if __name__ == "__main__":
    main()
