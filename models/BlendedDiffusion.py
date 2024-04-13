import os
import copy
import torch
from torch import nn
from diffusers.loaders import LoraLoaderMixin
from torch.nn.functional import cosine_similarity

from tqdm.notebook import tqdm

from models.StableDiffusion import StableDiffusion


class BlendedDiffusion(StableDiffusion):
    def __init__(self, model_path: str, lora_paths: list[str], train=False):
        super().__init__(model_path)
        
        self.unets = nn.ModuleList() if train else []
        for _ in lora_paths:
            new_unet = copy.deepcopy(self.unet.to(self.device))
            self.unets.append(new_unet)
        self.unet = None # let it be garbage collected since we don't need the original unet any more

        self.representative_embeddings = []
        
        # apply lora weights to each unet
        for unet, lora_path in zip(self.unets, lora_paths):
            # load lora weights
            state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora_path, weight_name='pytorch_lora_weights.safetensors')
            LoraLoaderMixin.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=unet)
            # load representative embedding
            representative_embedding = torch.load(os.path.join(lora_path, 'representative_embedding.pt'))
            # we do classifier-free guidance meaning the encoded text prompts has shape[0] of 2
            self.representative_embeddings.append(representative_embedding.repeat(2, 1, 1)) 
        
        # convert python list into tensor
        self.representative_embeddings = torch.stack(self.representative_embeddings).to(self.device)
        self.representative_embeddings = self.representative_embeddings.flatten(start_dim=1)

    
    def forward(self, prompt: str | list[str], num_inference_steps: int = 50):
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        
        encoded_prompt = self.encode_prompt(prompt)
        
        test_embedding = encoded_prompt.repeat(self.representative_embeddings.shape[0], 1, 1, 1)
        test_embedding = test_embedding.flatten(start_dim=1)
        similarities = cosine_similarity(test_embedding, self.representative_embeddings, dim=1)
        weights = torch.nn.functional.softmax(similarities, dim=0)
        weights = weights.view(-1, 1, 1, 1, 1)
        
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
            
            # due to memory constraints on gpu, we can't run these all in a single batch, so do it in a for loop
            noise_preds = []
            for unet in self.unets:
                noise_preds.append(self.predict_noise(latent_model_input, t, encoded_prompt, unet))
            noise_preds = torch.stack(noise_preds).to(self.device)
            
            # compute weighted average
            weighted_noise_preds = noise_preds * weights
            noise_pred = torch.sum(weighted_noise_preds, dim=0)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # decode the latent space to get generated image
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type='pil', do_denormalize=([True] * image.shape[0]))
        
        return image