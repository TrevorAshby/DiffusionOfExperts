import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.loaders import LoraLoaderMixin
from typing import Union

from tqdm.notebook import tqdm

import os


class StableDiffusion(nn.Module):
    def __init__(
        self,
        model_path: str,
        lora_path: Union[str, None] = None, # i.e. './model_downloads/clothes_finetuned_model'
        variant: Union[str, None] = None, # i.e. 'fp16' or 'bf16'
        device='cuda',
    ):
        super(StableDiffusion, self).__init__()
                
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        
        # models
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", variant=variant)
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(model_path, subfolder="vae", variant=variant)
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
        
        # make sure we don't compute unnecessary gradients
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False) # if there are lora weights, these will be finetuned instead
        
        self.representative_embedding = None
        
        # add in lora weights to unet if applicable
        if lora_path:
            # load lora weights
            state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora_path, weight_name='pytorch_lora_weights.safetensors')
            LoraLoaderMixin.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet)
            # load representative embedding
            self.representative_embedding = torch.load(os.path.join(lora_path, 'representative_embedding.pt'))
        
        self.device = device
        self.guidance_scale = 7.5
        self.sample_size: int = self.unet.config.sample_size
        self.vae_scale_factor: int = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def forward(self, prompt: Union[str, list[str]], num_inference_steps: int = 50):
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        
        encoded_prompt = self.encode_prompt(prompt)
        
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
        )
        
        # start denoising process
        for t in tqdm(timesteps):
            latent_model_input = torch.cat([latents] * 2) # double the latents since we have the unconditional text prompt too
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = self.predict_noise(latent_model_input, t, encoded_prompt)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # decode the latent space to get generated image
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type='pil', do_denormalize=([True] * image.shape[0]))
        
        return image

    def encode_prompt(self, prompt: Union[str, list[str]]):
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        prompt = self._maybe_convert_prompt(prompt, self.tokenizer)
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        
        # Now, do classifier-free guidance
        uncond_tokens = [""] * batch_size
        uncond_tokens = self._maybe_convert_prompt(uncond_tokens, self.tokenizer)
        max_uncond_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_uncond_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(self.device)
        negative_prompt_embeds = self.text_encoder(uncond_input_ids)[0]
        
        return torch.cat([negative_prompt_embeds, prompt_embeds])
        
    def prepare_latents(self, batch_size: int, num_channels_latents: int):
        height = self.sample_size * self.vae_scale_factor
        width = height
        
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        
        latents = randn_tensor(shape, device=self.device) * self.scheduler.init_noise_sigma
        return latents

    def predict_noise(self, latent_inputs: torch.Tensor, timestep: torch.Tensor, 
                      encoded_prompt: torch.Tensor, unet: Union[UNet2DConditionModel, None]=None) -> torch.Tensor:
        if unet is None:
            unet = self.unet
        # predict the noise residual
        noise_pred = unet(
            latent_inputs.to(unet.device),
            timestep.to(unet.device),
            encoder_hidden_states=encoded_prompt.to(unet.device),
            return_dict=False,
        )[0]
        
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        return noise_pred

    ## Huggingface implementations  ##

    def _maybe_convert_prompt(self, prompt: Union[str, list[str]], tokenizer: CLIPTokenizer):  
        r"""
        Processes prompts that include a special token corresponding to a multi-vector textual inversion embedding to
        be replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or if the textual inversion token is a single vector, the input prompt is returned.

        Parameters:
            prompt (`str` or list of `str`):
                The prompt or prompts to guide the image generation.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.

        Returns:
            `str` or list of `str`: The converted prompt
        """
        prompts = [prompt] if isinstance(prompt, str) else prompt
        prompts = [self._handle_multi_vector_textual_inversion(p, tokenizer) for p in prompts]

        if isinstance(prompt, str):
            return prompts[0]

        return prompts

    def _handle_multi_vector_textual_inversion(self, prompt: str, tokenizer: CLIPTokenizer):
        r"""
        Maybe convert a prompt into a "multi vector"-compatible prompt. If the prompt includes a token that corresponds
        to a multi-vector textual inversion embedding, this function will process the prompt so that the special token
        is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or a textual inversion token that is a single vector, the input prompt is simply returned.

        Parameters:
            prompt (`str`):
                The prompt to guide the image generation.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.

        Returns:
            `str`: The converted prompt
        """
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt