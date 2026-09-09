# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Intel Corporation. All rights reserved.
# --------------------------------------------------------------------------
"""Reference driver for the standalone Z-Image-Turbo ONNX export in `z-image-turbo-onnx/`
(see export_models.py).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import torch
from PIL import Image
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "z-image-turbo-onnx"

VAE_SCALING_FACTOR = 0.3611
VAE_SHIFT_FACTOR = 0.1159


def peak_memory_mb() -> float:
    mem_info = psutil.Process(os.getpid()).memory_info()
    # peak_wset is Windows-only; fall back to current RSS on Linux/macOS.
    peak_bytes = getattr(mem_info, "peak_wset", mem_info.rss)
    return peak_bytes / (1024 * 1024)


def input_dtype(session: ort.InferenceSession, name: str) -> np.dtype:
    onnx_type = next(i.type for i in session.get_inputs() if i.name == name)
    return np.float16 if onnx_type == "tensor(float16)" else np.float32


def select_webgpu_device(gpu: int) -> "ort.OrtEpDevice":
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "WebGpuExecutionProvider"]
    if not devices:
        raise RuntimeError("WebGPU requested but no WebGPU-capable device was found.")

    print("Available WebGPU devices:")
    for i, d in enumerate(devices):
        device_info = d.device
        print(f"  [{i}] {device_info.vendor}")
        print(f"      vendor_id: 0x{device_info.vendor_id:04x}")
        print(f"      device_id: 0x{device_info.device_id:04x}")
        for key, value in device_info.metadata.items():
            print(f"      {key}: {value}")

    if not 0 <= gpu < len(devices):
        raise ValueError(f"--gpu {gpu} out of range (found {len(devices)} WebGPU device(s)).")

    chosen = devices[gpu]
    print(f"Using WebGPU device [{gpu}]: {chosen.device.vendor}")
    return chosen


def log_session_io(label: str, session: ort.InferenceSession) -> None:
    print(f"{label}")
    print("  input:")
    for i in session.get_inputs():
        print(f"    {i.name}: {i.type} {i.shape}")
    print("  output:")
    for o in session.get_outputs():
        print(f"    {o.name}: {o.type} {o.shape}")


def to_uint8_hwc(vae_output: np.ndarray) -> np.ndarray:
    """vae_output: (1, 3, H, W) float, normalized to [-1, 1]."""
    chw = vae_output[0].astype(np.float32)
    chw = np.clip(chw * 0.5 + 0.5, 0.0, 1.0)
    return (chw * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)


def save_image(hwc: np.ndarray, path: str) -> None:
    Image.fromarray(hwc, mode="RGB").save(path)
    print(f"Image saved to {path} ({os.path.getsize(path) / 1024:.1f} KB)")


class Scheduler:
    """Flow-matching (shift=3) timestep schedule -- must match the sigma schedule baked into
    scheduler_step_model_f16.onnx (see build_helper_models.py)."""

    NUM_TRAIN_TIMESTEPS = 1000
    SHIFT = 3.0

    def timesteps(self, num_inference_steps: int) -> np.ndarray:
        sigma = np.linspace(1.0, self.NUM_TRAIN_TIMESTEPS, self.NUM_TRAIN_TIMESTEPS, dtype=np.float32)[::-1] / self.NUM_TRAIN_TIMESTEPS
        sigma_max, sigma_min = sigma[0], sigma[-1]

        t = np.linspace(sigma_max * self.NUM_TRAIN_TIMESTEPS, sigma_min * self.NUM_TRAIN_TIMESTEPS, num_inference_steps)
        sigmas = t / self.NUM_TRAIN_TIMESTEPS
        sigmas = self.SHIFT * sigmas / (1 + (self.SHIFT - 1) * sigmas)

        # Transformer's timestep is the reverse of the sigma schedule above (0=noise, 1=clean).
        timesteps = (self.NUM_TRAIN_TIMESTEPS - sigmas * self.NUM_TRAIN_TIMESTEPS) / self.NUM_TRAIN_TIMESTEPS
        timesteps[-1] = 1.0
        return timesteps.astype(np.float32)


class ZImagePipeline:
    def __init__(self, model_dir: str, ep: str, gpu: int = 0, sync: bool = False, safety_checker: bool = False):
        self.sync = sync
        self.use_safety_checker = safety_checker
        available = ort.get_available_providers()
        use_webgpu = ep == "WebGPU" or (not ep and "WebGpuExecutionProvider" in available)
        if ep == "WebGPU" and "WebGpuExecutionProvider" not in available:
            raise RuntimeError("WebGPU requested but not available in this onnxruntime build.")

        if use_webgpu:
            print("Execution provider: WebGpuExecutionProvider")
            webgpu_device = select_webgpu_device(gpu)
            sess_options = ort.SessionOptions()
            sess_options.add_provider_for_devices([webgpu_device], {})
            session_kwargs = {"sess_options": sess_options}
            self.device_type, self.device_id = "webgpu", gpu
        else:
            print("Execution provider: CPUExecutionProvider")
            session_kwargs = {"providers": ["CPUExecutionProvider"]}
            self.device_type, self.device_id = "cpu", 0

        onnx_dir = os.path.join(model_dir, "onnx")
        self.text_encoder = ort.InferenceSession(os.path.join(onnx_dir, "text_encoder_model_q4f16.onnx"), **session_kwargs)
        self.transformer = ort.InferenceSession(os.path.join(onnx_dir, "transformer_model_q4f16.onnx"), **session_kwargs)
        self.scheduler_step = ort.InferenceSession(os.path.join(onnx_dir, "scheduler_step_model_f16.onnx"), **session_kwargs)
        self.vae_pre_process = ort.InferenceSession(os.path.join(onnx_dir, "vae_pre_process_model_f16.onnx"), **session_kwargs)
        self.vae_decoder = ort.InferenceSession(os.path.join(onnx_dir, "vae_decoder_model_f16.onnx"), **session_kwargs)

        self.text_encoder_iob = self.text_encoder.io_binding()
        self.transformer_iob = self.transformer.io_binding()
        self.scheduler_step_iob = self.scheduler_step.io_binding()
        self.vae_pre_process_iob = self.vae_pre_process.io_binding()
        self.vae_decoder_iob = self.vae_decoder.io_binding()

        print("Model shapes:")
        log_session_io("text_encoder", self.text_encoder)
        log_session_io("transformer", self.transformer)
        log_session_io("scheduler_step", self.scheduler_step)
        log_session_io("vae_pre_process", self.vae_pre_process)
        log_session_io("vae_decoder", self.vae_decoder)

        self.transformer_dtype = input_dtype(self.transformer, "hidden_states")
        self.scheduler_step_dtype = input_dtype(self.scheduler_step, "latents")
        self.vae_pre_process_dtype = input_dtype(self.vae_pre_process, "latents")
        self.vae_decoder_dtype = input_dtype(self.vae_decoder, "latent_sample")

        self.text_encoder_output = self.text_encoder.get_outputs()[0].name
        self.transformer_output = self.transformer.get_outputs()[0].name
        self.scheduler_step_output = self.scheduler_step.get_outputs()[0].name
        self.vae_pre_process_output = self.vae_pre_process.get_outputs()[0].name
        self.vae_decoder_output = self.vae_decoder.get_outputs()[0].name

        if self.use_safety_checker:
            sc_prep_path = os.path.join(onnx_dir, "sc_prep_model_f16.onnx")
            safety_checker_path = os.path.join(onnx_dir, "safety_checker_model_f16.onnx")
            for path in (sc_prep_path, safety_checker_path):
                if not os.path.isfile(path):
                    raise SystemExit(
                        f"--safety_checker requested but {path} is missing -- build it with "
                        "build_safety_checker.py (see export_models.py's "
                        "--safety_checker_checkpoint)."
                    )
            self.sc_prep = ort.InferenceSession(sc_prep_path, **session_kwargs)
            self.safety_checker = ort.InferenceSession(safety_checker_path, **session_kwargs)
            self.sc_prep_iob = self.sc_prep.io_binding()
            self.safety_checker_iob = self.safety_checker.io_binding()
            log_session_io("sc_prep", self.sc_prep)
            log_session_io("safety_checker", self.safety_checker)
            self.sc_prep_dtype = input_dtype(self.sc_prep, "sample")
            self.safety_checker_dtype = input_dtype(self.safety_checker, "clip_input")
            self.sc_prep_output = self.sc_prep.get_outputs()[0].name
            self.safety_checker_output = self.safety_checker.get_outputs()[0].name

        # GPU EP only: uploads prompt embeds to the device once per prompt.
        self.embeds_uploader = None
        if self.device_type != "cpu":
            self._build_embeds_uploader(session_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.scheduler = Scheduler()

    def _build_embeds_uploader(self, session_kwargs: dict) -> None:
        from onnx import helper, TensorProto

        elem_type = TensorProto.FLOAT16 if self.transformer_dtype == np.float16 else TensorProto.FLOAT
        value_in = helper.make_tensor_value_info("input", elem_type, None)
        value_out = helper.make_tensor_value_info("output", elem_type, None)
        graph = helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"])], "upload_embeds", [value_in], [value_out]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10

        self.embeds_uploader = ort.InferenceSession(model.SerializeToString(), **session_kwargs)
        self.embeds_uploader_iob = self.embeds_uploader.io_binding()
        self.embeds_uploader_output = self.embeds_uploader.get_outputs()[0].name

    def to_ort_value(self, array: np.ndarray) -> ort.OrtValue:
        """Always backs a CPU OrtValue; GPU residency instead goes through bind_output."""
        return ort.OrtValue.ortvalue_from_numpy(array)

    def _run_bound(
        self, label: str, session: ort.InferenceSession, iob: "ort.IOBinding", output_names: list, inputs: dict
    ) -> list:
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                iob.bind_cpu_input(name, value)
            else:
                iob.bind_ortvalue_input(name, value)
        for name in output_names:
            iob.bind_output(name, device_type=self.device_type, device_id=self.device_id)
        if self.sync:
            iob.synchronize_inputs()
        session.run_with_iobinding(iob)
        outputs = iob.get_outputs()
        if self.sync:
            # Reading each output back forces the real GPU wait (WebGPU dispatch is async).
            iob.synchronize_outputs()
            for value in outputs:
                value.numpy()
        self._log_io(label, inputs, output_names, outputs)
        return outputs

    def _device_label(self, value: ort.OrtValue) -> str:
        # device_name() reports "cuda" for any non-CPU device, not just actual CUDA.
        name = value.device_name()
        return self.device_type if name == "cuda" else name

    def _log_io(self, label: str, inputs: dict, output_names: list, outputs: list) -> None:
        print(f"  [{label}] io:")
        for name, value in inputs.items():
            device = self._device_label(value) if isinstance(value, ort.OrtValue) else "cpu"
            print(f"    in  {name}: {device}")
        for name, value in zip(output_names, outputs):
            print(f"    out {name}: {self._device_label(value)}")

    def encode_prompt(self, prompt: str) -> ort.OrtValue:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="np")
        input_ids = inputs.input_ids.astype(np.int64)
        attention_mask = inputs.attention_mask.astype(np.int64)

        seq_len = int(attention_mask.sum())
        print(f"Tokenized prompt: {seq_len} tokens")

        embeds_ov = self._run_bound(
            "text_encoder", self.text_encoder, self.text_encoder_iob, [self.text_encoder_output],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]

        # Slicing to prompt length + padding to a multiple of 32 (transformer has no mask/padding
        # logic, see build_transformer.py); a one-time per-prompt host round-trip is fine here.
        embeds = embeds_ov.numpy()[:, :seq_len, :]
        pad_len = (-embeds.shape[1]) % 32
        if pad_len:
            pad = np.repeat(embeds[:, -1:, :], pad_len, axis=1)
            embeds = np.concatenate([embeds, pad], axis=1)

        embeds = embeds.astype(self.transformer_dtype)
        if self.embeds_uploader is None:
            embeds_ov = self.to_ort_value(embeds)
        else:
            embeds_ov = self._run_bound(
                "upload_embeds", self.embeds_uploader, self.embeds_uploader_iob,
                [self.embeds_uploader_output], {"input": embeds},
            )[0]
        print(f"prompt_embeds device: {self._device_label(embeds_ov)}")
        return embeds_ov

    def run(
        self,
        prompt: str,
        output_path: str,
        num_inference_steps: int,
        height: int,
        width: int,
        all_images: bool = False,
        seed: int = 42,
    ) -> None:
        print(f"height: {height}, width: {width}, steps: {num_inference_steps}")

        latent_h, latent_w = height // 8, width // 8
        latents = torch.randn(
            (1, 16, latent_h, latent_w), generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32
        ).numpy()
        latents_ov = self.to_ort_value(latents.astype(self.transformer_dtype))
        print(f"latents device: {latents_ov.device_name()}")
        timesteps = self.scheduler.timesteps(num_inference_steps)

        total_ms = 0.0

        def timed(label, fn, *args):
            nonlocal total_ms
            start = time.perf_counter()
            result = fn(*args)
            ms = (time.perf_counter() - start) * 1000
            if self.sync:  # per-call times are only meaningful with --sync
                print(f"{label} time: {ms:.2f} ms")
            total_ms += ms
            return result

        print(f"Prompt:\n{prompt}")
        prompt_embeds = timed("text_encoder", self.encode_prompt, prompt)

        for step in range(num_inference_steps):
            timestep = timesteps[step]
            if not self.sync:
                print(f"step {step}/{num_inference_steps}, timestep {timestep:.4f}")

            noise_pred_ov = timed(f"transformer-{step}", self._run_transformer, latents_ov, timestep, prompt_embeds)
            latents_ov = timed(f"scheduler_step-{step}", self._run_scheduler_step, noise_pred_ov, latents_ov, step, num_inference_steps)

            if all_images and step < num_inference_steps - 1:
                path = Path(output_path)
                hwc = to_uint8_hwc(self._decode(latents_ov).numpy())
                save_image(hwc, str(path.with_name(f"{path.stem}-step{step}{path.suffix}")))

        scaled_latents_ov = timed("vae_pre_process", self._run_vae_pre_process, latents_ov)
        image_ov = timed("vae_decoder", self._run_vae_decoder, scaled_latents_ov)

        def _postprocess():
            # image_ov.numpy() forces the final GPU sync; keep it inside the timed window.
            return to_uint8_hwc(image_ov.numpy())

        hwc = timed("postprocess", _postprocess)
        save_image(hwc, output_path)
        print(f"total time: {total_ms:.2f} ms")

        if self.use_safety_checker:
            start = time.perf_counter()
            nsfw = self._run_safety_checker(image_ov)
            ms = (time.perf_counter() - start) * 1000
            print(f"safety_checker time (excluded from total): {ms:.2f} ms")
            print(f"Safety Checker - NSFW: {'Yes' if nsfw else 'No'}")

    def _run_transformer(self, latents_ov: ort.OrtValue, timestep: float, prompt_embeds_ov: ort.OrtValue) -> ort.OrtValue:
        inputs = {
            "hidden_states": latents_ov,
            "timestep": np.array([timestep], dtype=self.transformer_dtype),
            "encoder_hidden_states": prompt_embeds_ov,
        }
        return self._run_bound("transformer", self.transformer, self.transformer_iob, [self.transformer_output], inputs)[0]

    def _run_scheduler_step(
        self, noise_pred_ov: ort.OrtValue, latents_ov: ort.OrtValue, step: int, num_inference_steps: int
    ) -> ort.OrtValue:
        inputs = {
            "noise_pred": noise_pred_ov,
            "latents": latents_ov,
            "step_info": np.array([step, num_inference_steps], dtype=self.scheduler_step_dtype),
        }
        return self._run_bound(
            "scheduler_step", self.scheduler_step, self.scheduler_step_iob, [self.scheduler_step_output], inputs
        )[0]

    def _run_vae_pre_process(self, latents_ov: ort.OrtValue) -> ort.OrtValue:
        return self._run_bound(
            "vae_pre_process", self.vae_pre_process, self.vae_pre_process_iob, [self.vae_pre_process_output],
            {"latents": latents_ov},
        )[0]

    def _run_vae_decoder(self, scaled_latents_ov: ort.OrtValue) -> ort.OrtValue:
        if self.vae_pre_process_dtype != self.vae_decoder_dtype:
            scaled_latents_ov = self.to_ort_value(scaled_latents_ov.numpy().astype(self.vae_decoder_dtype))
        return self._run_bound(
            "vae_decoder", self.vae_decoder, self.vae_decoder_iob, [self.vae_decoder_output],
            {"latent_sample": scaled_latents_ov},
        )[0]

    def _decode(self, latents_ov: ort.OrtValue) -> ort.OrtValue:
        return self._run_vae_decoder(self._run_vae_pre_process(latents_ov))

    def _run_safety_checker(self, image_ov: ort.OrtValue) -> bool:
        # "sample" is the raw [-1,1] VAE output, not the postprocessed uint8 image.
        if self.vae_decoder_dtype != self.sc_prep_dtype:
            image_ov = self.to_ort_value(image_ov.numpy().astype(self.sc_prep_dtype))
        clip_input_ov = self._run_bound(
            "sc_prep", self.sc_prep, self.sc_prep_iob, [self.sc_prep_output], {"sample": image_ov}
        )[0]
        if self.sc_prep_dtype != self.safety_checker_dtype:
            clip_input_ov = self.to_ort_value(clip_input_ov.numpy().astype(self.safety_checker_dtype))
        has_nsfw_ov = self._run_bound(
            "safety_checker", self.safety_checker, self.safety_checker_iob, [self.safety_checker_output],
            {"clip_input": clip_input_ov},
        )[0]
        return bool(has_nsfw_ov.numpy().ravel()[0])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Z-Image-Turbo inference against the standalone ONNX export in z-image-turbo-onnx/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model", nargs="?", default=str(DEFAULT_MODEL_DIR),
        help="Path to the exported ONNX model directory (containing onnx/ and tokenizer/, as written by export_models.py).",
    )
    parser.add_argument("--ep", default="", choices=["WebGPU", "CPU"], help="Execution provider.")
    parser.add_argument("--gpu", type=int, default=0, help="WebGPU device index, for machines with more than one GPU. Ignored with --ep CPU.")
    parser.add_argument(
        "--prompt",
        default="In a tranquil garden at dusk, a young Chinese woman stands gracefully in a red Hanfu with gold embroidery. Her flawless complexion features a red floral pattern on her forehead, enhancing her warm smile and expressive eyes. With her hair styled in a high bun adorned with a golden phoenix headdress, she holds a round folding fan decorated with nature scenes. Cherry blossom trees surround her, their petals drifting in the breeze, while a silhouetted pagoda (西安大雁塔) adds depth, blending tradition with modernity.",
        help="Text prompt to generate the image from.",
    )
    parser.add_argument("-s", "--step", type=int, default=4, help="Number of denoising steps.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("-o", "--output_name", default="z-image-turbo.png", help="Output image path.")
    parser.add_argument("-l", "--loop", type=int, default=1, help="Number of times to repeat generation (for benchmarking).")
    parser.add_argument("-a", "--all_images", action="store_true", help="Also write an image after every denoising step.")
    parser.add_argument(
        "--sync", action="store_true",
        help="Force a GPU sync after every model call for accurate per-call timing (default: off; "
        "WebGPU dispatch is otherwise async, so per-call times would only reflect submission "
        "overhead, not real compute time).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Latent noise seed.")
    parser.add_argument(
        "--safety_checker", action="store_true",
        help="Run the optional NSFW safety checker (sc_prep + safety_checker_model_f16.onnx, "
        "~580 MB extra; build both with build_safety_checker.py first). Its runtime is printed "
        "separately and is NOT included in the pipeline's total-time metric.",
    )
    return parser.parse_args()


def main():
    # Windows consoles default stdout to the system codepage (e.g. cp1252), which can't encode
    # prompts containing non-Latin-1 characters (the default prompt has Chinese in it).
    sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(f"Model path not found: {args.model}")
    if not os.path.isdir(os.path.join(args.model, "tokenizer")):
        raise SystemExit(
            f"Tokenizer not found under {args.model}/tokenizer -- re-run export_models.py to "
            "populate it."
        )

    pipeline = ZImagePipeline(args.model, args.ep, args.gpu, args.sync, args.safety_checker)

    output_name = Path(args.output_name)
    stem = f"{output_name.stem}_{args.width}x{args.height}_steps{args.step}"
    for i in range(args.loop):
        loop_name = output_name.with_name(f"{stem}_loop{i}{output_name.suffix}")
        pipeline.run(args.prompt, str(loop_name), args.step, args.height, args.width, args.all_images, args.seed)

    print(f"Peak Memory: {peak_memory_mb():.2f} MB")


if __name__ == "__main__":
    main()
