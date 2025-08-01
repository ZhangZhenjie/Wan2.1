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
        low_vram_mode=False  # Only affects self.model
    ):
        r"""
        Initializes the image-to-video generation model components.
        Low VRAM mode only applies to the main model (self.model).
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.low_vram_mode = low_vram_mode
        self._compiled = False

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype  # Keep original dtype for non-model components

        shard_fn = partial(shard_model, device_id=device_id)
        
        # Initialize text encoder (T5) - always kept in original precision
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

        # Initialize VAE - always kept in original precision
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        # Initialize CLIP - always kept in original precision
        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        # Initialize main model
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        # Handle distributed/parallel settings
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

        # Handle model placement
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        # Apply low VRAM mode only to self.model if requested
        if low_vram_mode:
            self._init_model_low_vram_mode()

        self.sample_neg_prompt = config.sample_neg_prompt

    def _init_model_low_vram_mode(self):
        """Initialize only self.model for low VRAM mode."""
        logging.info("Initializing low VRAM mode for main model only")
        
        # Set model to eval and half precision
        self.model.eval()
        self.model.half()
        
        # Enable memory-efficient attention
        try:
            self.model.enable_xformers_memory_efficient_attention()
            logging.info("Enabled xformers memory efficient attention for main model")
        except Exception as e:
            logging.warning(f"Couldn't enable xformers for main model: {e}")
        
        # Compile model if available
        if hasattr(torch, "compile") and not self._compiled:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
                logging.info("Main model compiled with torch.compile()")
            except Exception as e:
                logging.warning(f"torch.compile failed for main model: {e}")
        
        torch.cuda.empty_cache()

    def _ensure_model_low_vram_mode(self):
        """Ensure main model is in low VRAM mode."""
        if self.low_vram_mode:
            if self.model.dtype != torch.float16:
                self.model.half()
            self.model.eval()

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
                 offload_model=True):
        """
        Generates video frames from input image and text prompt.
        Only the main model operates in low VRAM mode when enabled.
        """
        # Ensure main model is in correct mode
        if self.low_vram_mode:
            self._ensure_model_low_vram_mode()

        # Prepare input image (always full precision)
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        # Calculate dimensions
        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
                      self.patch_size[1] * self.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
                      self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        # Calculate sequence length
        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # Prepare noise (use model's dtype for noise)
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise_dtype = torch.float16 if self.low_vram_mode else self.param_dtype
        noise = torch.randn(
            16, (F - 1) // 4 + 1, lat_h, lat_w,
            dtype=noise_dtype,
            generator=seed_g,
            device=self.device
        )

        # Prepare mask (use model's dtype for mask)
        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device, dtype=noise_dtype)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

        # Handle negative prompt
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Prepare text embeddings (always full precision)
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

        # Prepare CLIP features (always full precision)
        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()
            torch.cuda.empty_cache()

        # Prepare input tensor (always full precision)
        input_tensor = torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
            torch.zeros(3, F - 1, h, w)
        ], dim=1).to(self.device)

        # Encode with VAE (always full precision)
        y = self.vae.encode([input_tensor])[0]
        y = torch.concat([msk, y])

        # Prepare context managers
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # Main generation loop
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # Initialize scheduler
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latent = noise

            # Prepare arguments
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

            # Ensure model is on the right device
            if offload_model:
                torch.cuda.empty_cache()
            self.model.to(self.device)

            # Process in chunks for memory efficiency
            x0 = None
            chunk_size = 8  # Adjust based on available memory

            for i in range(0, len(timesteps), chunk_size):
                t_chunk = timesteps[i:i + chunk_size]
                for t in t_chunk:
                    # Prepare inputs (convert to model's dtype)
                    latent_model_input = [latent.to(self.device, non_blocking=True)]
                    timestep = torch.tensor([t], device=self.device)

                    if self.low_vram_mode:
                        latent_model_input = [x.half() for x in latent_model_input]
                        timestep = timestep.half()
                        arg_c['clip_fea'] = arg_c['clip_fea'].half()
                        arg_c['context'] = [c.half() for c in arg_c['context']]
                        arg_null['context'] = [c.half() for c in arg_null['context']]

                    # Forward pass with conditioning
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    if offload_model:
                        noise_pred_cond = noise_pred_cond.to('cpu')

                    # Forward pass without conditioning
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                    if offload_model:
                        noise_pred_uncond = noise_pred_uncond.to('cpu')

                    # Combine predictions
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                    if offload_model:
                        noise_pred = noise_pred.to(self.device)
                        latent = latent.to(self.device)

                    # Scheduler step
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                        return_dict=False, generator=seed_g)[0]

                    latent = temp_x0.squeeze(0)
                    x0 = [latent.to(self.device)]

                    # Clean up
                    del latent_model_input, timestep, noise_pred_cond, noise_pred_uncond, noise_pred
                    if self.low_vram_mode:
                        torch.cuda.empty_cache()
                        gc.collect()

            # Decode video if rank 0 (always full precision)
            if self.rank == 0:
                self.vae.model.to(self.device)
                videos = self.vae.decode(x0)
                if offload_model:
                    self.vae.model.cpu()

        # Final cleanup
        if offload_model:
            self.model.cpu()
        torch.cuda.empty_cache()
        gc.collect()

        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
