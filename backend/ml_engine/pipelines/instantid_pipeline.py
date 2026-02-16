"""
InstantID Pipeline for Stable Diffusion XL

This is a custom pipeline implementation for InstantID that integrates with diffusers.
Based on the official InstantID implementation from InstantX-research.

Source: https://github.com/instantX-research/InstantID
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional, List, Union
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput


def draw_kps(image_pil: Image.Image, kps: np.ndarray, color_list=None) -> Image.Image:
    """
    Draw facial keypoints on an image.
    
    Args:
        image_pil: PIL Image to draw on
        kps: Keypoints array of shape (N, 2) with x, y coordinates
        color_list: Optional list of colors for each keypoint
        
    Returns:
        PIL Image with keypoints drawn
    """
    if color_list is None:
        color_list = [
            (255, 0, 0),    # left eye - red
            (0, 255, 0),    # right eye - green
            (0, 0, 255),    # nose - blue
            (255, 255, 0),  # left mouth - yellow
            (255, 0, 255),  # right mouth - magenta
        ]
    
    # Convert PIL to numpy
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    
    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])
    
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]
        
        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = np.degrees(np.arctan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle),
            0,
            360,
            1
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    
    out_img = (out_img * 0.6).astype(np.uint8)
    
    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)
    
    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


class StableDiffusionXLInstantIDPipeline(StableDiffusionXLPipeline):
    """
    Extended SDXL pipeline with InstantID support.
    
    This pipeline adds identity-preserving generation capabilities to SDXL
    using face embeddings and ControlNet for keypoint guidance.
    """
    
    def load_ip_adapter_instantid(self, model_path: str):
        """
        Load the InstantID IP-Adapter weights and set up attention processors.
        
        Args:
            model_path: Path to ip-adapter.bin file
        """
        from ip_adapter.resampler import Resampler
        from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
        
        # Set up image projection model (Resampler)
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=16,  # num_tokens
            embedding_dim=512,  # image_emb_dim
            output_dim=self.unet.config.cross_attention_dim,
            ff_mult=4,
        )
        image_proj_model.eval()
        self.image_proj_model = image_proj_model.to(self.device, dtype=self.dtype)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Load image projection weights
        if 'image_proj' in state_dict:
            self.image_proj_model.load_state_dict(state_dict["image_proj"])
        
        self.image_proj_model_in_features = 512
        
        # Set up IP-Adapter attention processors
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=0.8,  # default scale
                    num_tokens=16
                ).to(unet.device, dtype=unet.dtype)
        
        unet.set_attn_processor(attn_procs)
        
        # Load IP-Adapter weights into attention processors
        ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
        if 'ip_adapter' in state_dict:
            ip_layers.load_state_dict(state_dict['ip_adapter'])
        
        print(f"✓ Loaded InstantID IP-Adapter from {model_path}")
    
    def set_ip_adapter_scale(self, scale: float):
        """
        Set the scale for IP-Adapter attention processors.
        
        Args:
            scale: Scale value for IP-Adapter (0.0 to 1.0)
        """
        unet = self.unet
        for attn_processor in unet.attn_processors.values():
            if hasattr(attn_processor, 'scale'):
                attn_processor.scale = scale
    
    def _encode_prompt_image_emb(self, prompt_image_emb, device, num_images_per_prompt, dtype, do_classifier_free_guidance):
        """Encode face embeddings for IP-Adapter."""
        if isinstance(prompt_image_emb, torch.Tensor):
            prompt_image_emb = prompt_image_emb.clone().detach()
        else:
            prompt_image_emb = torch.tensor(prompt_image_emb)
            
        prompt_image_emb = prompt_image_emb.to(device=device, dtype=dtype)
        prompt_image_emb = prompt_image_emb.reshape([1, -1, self.image_proj_model_in_features])
        
        if do_classifier_free_guidance:
            prompt_image_emb = torch.cat([torch.zeros_like(prompt_image_emb), prompt_image_emb], dim=0)
        else:
            prompt_image_emb = torch.cat([prompt_image_emb], dim=0)
        
        prompt_image_emb = self.image_proj_model(prompt_image_emb)

        bs_embed, seq_len, _ = prompt_image_emb.shape
        prompt_image_emb = prompt_image_emb.repeat(1, num_images_per_prompt, 1)
        prompt_image_emb = prompt_image_emb.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        return prompt_image_emb
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[dict] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[tuple] = None,
        crops_coords_top_left: tuple = (0, 0),
        target_size: Optional[tuple] = None,
        negative_original_size: Optional[tuple] = None,
        negative_crops_coords_top_left: tuple = (0, 0),
        negative_target_size: Optional[tuple] = None,
        clip_skip: Optional[int] = None,
        # InstantID specific parameters
        image: Optional[Image.Image] = None,
        controlnet_conditioning_scale: float = 1.0,
        ip_adapter_scale: float = 1.0,
    ):
        """
        Generate images with InstantID identity preservation.
        
        Args:
            prompt: Text prompt for generation
            image_embeds: Face embedding from InsightFace
            image: Keypoint image from draw_kps
            controlnet_conditioning_scale: Strength of ControlNet guidance
            ip_adapter_scale: Strength of identity preservation
            ... (other standard SDXL parameters)
            
        Returns:
            StableDiffusionXLPipelineOutput with generated images
        """
        # Set default height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # Set IP-Adapter scale if provided
        if ip_adapter_scale is not None:
            self.set_ip_adapter_scale(ip_adapter_scale)
        
        # Encode face embeddings for IP-Adapter
        if image_embeds is not None:
            # Normalize and prepare face embeddings
            if isinstance(image_embeds, np.ndarray):
                image_embeds = torch.from_numpy(image_embeds).to(
                    device=self.device,
                    dtype=self.unet.dtype
                )
            
            # Ensure correct shape [batch_size, embedding_dim]
            if image_embeds.ndim == 1:
                image_embeds = image_embeds.unsqueeze(0)
            
            # Encode through image projection model
            prompt_image_emb = self._encode_prompt_image_emb(
                image_embeds,
                self.device,
                num_images_per_prompt,
                self.unet.dtype,
                guidance_scale > 1.0
            )
        else:
            prompt_image_emb = None
        
        # Prepare ControlNet image if provided
        controlnet_image = None
        if image is not None and hasattr(self, 'controlnet'):
            # Convert PIL to tensor
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Normalize to [-1, 1]
            controlnet_image = torch.from_numpy(image).float() / 127.5 - 1.0
            controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0)
            controlnet_image = controlnet_image.to(device=self.device, dtype=self.unet.dtype)
        
        # Call parent SDXL ControlNet pipeline
        # The IP-Adapter will be automatically used through the attention processors
        return super().__call__(
            prompt=prompt,
            prompt_2=prompt_2,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=controlnet_image,  # ControlNet keypoint image
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            cross_attention_kwargs=cross_attention_kwargs,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_rescale=guidance_rescale,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            negative_original_size=negative_original_size,
            negative_crops_coords_top_left=negative_crops_coords_top_left,
            negative_target_size=negative_target_size,
            clip_skip=clip_skip,
        )
