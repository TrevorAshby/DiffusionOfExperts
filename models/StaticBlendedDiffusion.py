import torch
from torch import nn

from tqdm.notebook import tqdm

from models.BlendedDiffusion import BlendedDiffusion


class StaticBlendedDiffusion(BlendedDiffusion):
    def __init__(self, model_path: str, lora_paths: list[str]):
        super().__init__(model_path, lora_paths, train=True)
        
        self.representative_embeddings = None # let it be garbage collected
        self.weights = nn.Parameter(torch.rand(len(lora_paths)))

    
    def forward(self, prompt: str | list[str], num_inference_steps: int = 50):
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
            
            # due to memory constraints on gpu, we can't run these all in a single batch, so do it in a for loop
            noise_preds = []
            for unet in self.unets:
                noise_preds.append(self.predict_noise(latent_model_input, t, encoded_prompt, unet))
            noise_preds = torch.stack(noise_preds).to(self.device)
            
            # compute weighted average
            weighted_noise_preds = noise_preds * self.weights.view(-1, 1, 1, 1, 1)
            noise_pred = torch.sum(weighted_noise_preds, dim=0)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # decode the latent space to get generated image
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type='pil', do_denormalize=([True] * image.shape[0]))
        
        return image