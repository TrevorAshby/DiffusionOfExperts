import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from typing import Union, List


class ExpertDiffuser(nn.Module):
    def __init__(self, path: str, quantized=False):
        self.quantized = quantized
        self.pipe = StableDiffusionPipeline.from_pretrained(path)
    
    def forward(self, prompts: Union[str, List[str]]):
        if self.quantized:
            pass
        else:
            return self.pipe(prompts)
        

class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        # experts
        self.clothes_expert = ExpertDiffuser(config['clothes']['path'], quantized=config['clothes']['quantized'])
        self.pixelart_expert = ExpertDiffuser(config['pixelart']['path'], quantized=config['pixelart']['quantized'])
        self.photographs_expert = ExpertDiffuser(config['photographs']['path'], quantized=config['photographs']['quantized'])
        
        # selection network
        self.selection_network = nn.Sequential()
    
    def forward(self, prompts: Union[str, List[str]]):
        clothes_images = self.clothes_expert(prompts)
        pixelart_images = self.pixelart_expert(prompts)
        photograhs_images = self.photographs_expert(prompts)
        
        inputs_to_selection_network = torch.cat((clothes_images, pixelart_images, photograhs_images), dim=1)
        return self.selection_network(inputs_to_selection_network)
        
        