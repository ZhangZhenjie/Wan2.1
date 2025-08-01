# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        low_vram_mode=False  # ✅ New flag
    ):
        r"""
        Initializes the image-to-video generation model components.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.low_vram_mode = low_vram_mode

        self.num_train_timesteps = config.num_train_timesteps

        # Use half precision if low_vram_mode
        if low_vram_mode:
            self.param_dtype = torch.float16
        else:
            self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from .distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        # ✅ Apply memory-efficient settings if requested
        if low_vram_mode:
            self.model.eval()
            self.model.half()
            self.vae.model.eval()
            self.vae.model.half()
            self.clip.model.eval()
            self.clip.model.half()

            try:
                self.model.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            # Compile model once (PyTorch 2.0+)
            if hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    self._compiled = True
                except Exception as e:
                    logging.warning(f"torch.compile failed: {e}")
            torch.cuda.empty_cache()

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 low_vram_mode=True):
        """
        Generates video frames from input image and text prompt using diffusion process.
        """
        import gc
        from contextlib import contextmanager

        if low_vram_mode:
            if not getattr(self, "_compiled", False) and hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    self._compiled = True
                except Exception as e:
                    print(f"[WARN] torch.compile failed: {e}")

            self.model.eval()
            self.clip.model.eval()
            self.vae.model.eval()
            self.param_dtype = torch.float16
            self.model.half()
            self.clip.model.half()
            self.vae.model.half()
            try:
                self.model.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            torch.cuda.empty_cache()

        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
                      self.patch_size[1] * self.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
                      self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(16, (F - 1) // 4 + 1, lat_h, lat_w, dtype=torch.float16 if low_vram_mode else torch.float32, generator=seed_g, device=self.device)

        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
                torch.cuda.empty_cache()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()
            torch.cuda.empty_cache()

        input_tensor = torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
            torch.zeros(3, F - 1, h, w)
        ], dim=1).to(self.device)

        y = self.vae.encode([input_tensor])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()
            self.model.to(self.device)

            x0 = None
            chunk_size = 8

            for i in range(0, len(timesteps), chunk_size):
                t_chunk = timesteps[i:i + chunk_size]
                for t in t_chunk:
                    latent_model_input = [latent.to(self.device, non_blocking=True)]
                    timestep = torch.tensor([t], device=self.device)

                    latent_model_input = [x.to(self.param_dtype) for x in latent_model_input]
                    timestep = timestep.to(self.param_dtype)

                    arg_c['clip_fea'] = arg_c['clip_fea'].to(self.param_dtype)
                    arg_c['context'] = [c.to(self.param_dtype) for c in arg_c['context']]
                    arg_null['context'] = [c.to(self.param_dtype) for c in arg_null['context']]

                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                    if offload_model:
                        torch.cuda.empty_cache()

                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                    if offload_model:
                        torch.cuda.empty_cache()

                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                    latent = latent.to(torch.device('cpu') if offload_model else self.device)

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                        return_dict=False, generator=seed_g)[0]

                    latent = temp_x0.squeeze(0)
                    x0 = [latent.to(self.device)]

                    del latent_model_input, timestep, noise_pred_cond, noise_pred_uncond, noise_pred
                    if low_vram_mode:
                        torch.cuda.empty_cache()
                        gc.collect()

            if offload_model or low_vram_mode:
                self.model.cpu()
                self.clip.model.cpu()
                self.vae.model.cpu()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, sample_scheduler
        if offload_model or low_vram_mode:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
