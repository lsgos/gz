import torch
import torch.nn as nn
import math
import torch
from torch.nn import functional as F
from torch import nn
from torch.distributions import constraints
from torchvision.datasets import MNIST
from torchvision import transforms as tvt
from neural_networks.layers import ConvBlock, UpResBloc, Reshape
import pyro
from pyro import infer, optim, poutine
from pyro import distributions as D
from galaxy_gen.forward_models import random_pose_transform
from galaxy_gen.backward_models import delta_sample_transformer_params
from galaxy_gen.etn import transformers, networks
from galaxy_gen.etn import transforms as T
from galaxy_gen.etn import coordinates

from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import glob


class Encoder(nn.Module):
    """
    Will take any insize as long as it is divisible by 8
    output is a dictionary with four variables
    dictionary["transform"] is a transformation
    dctionary["transform_params"] are parameters that parameterise that 
    transformation
    dictionary["z_mu"] is the mean of the latent variable
    dictionary["std_mu"] is the standard dev of that latent variable
    so the latent space is the z and std in this transformed coordinate frame
    We output the theta that parameterises this coordinate frame
    and the z in this transformed coordinate frame

    """
    def __init__(self,
                 transformer: transformers.Transformer,
                 insize=56, z_dim=10, spatial_vae=False, num_pose_params_dict=None):
        super().__init__()
        self.transformer = transformer
        self.insize = insize
        self.spatial_vae = spatial_vae
        self.linear_size = int(((insize/16)**2)*16)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # second conv pair
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # third conv
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # fourth conv
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # linear

            # linear

        )
        # layers for latents
        self.loc = nn.Linear(int(self.linear_size), z_dim)
        self.scale = nn.Linear(int(self.linear_size), z_dim)
        self.relu = nn.ReLU()

        self.num_transforms = len(self.transformer.transformers)
        self.parameters_per_transform = []
        for transform in self.transformer.transformers:
            self.parameters_per_transform.append(transform.num_outputs)
        self.layer_for_params = nn.Linear(int(self.linear_size), sum(self.parameters_per_transform))

    def forward(self, x):
        output = {}
        x = x - 0.222 # lol this complicates switching datasets severely
        x = x / 0.156
        
        if not self.spatial_vae:
            transform_output = self.transformer(x)
            # dont transform input with a spatial vae
            output["transform"] = transform_output["transform"]
            output["transform_params"] = transform_output["params"]
            # the above is a list of len = number of transforms
            # each of those consists of a tuple of transforms, len 1 or 2 depending on type of transform
            # each of those is a torch.tensor, depen ding on batch_size

            grid = coordinates.identity_grid(
                [self.insize, self.insize], device=x.device
            )
            grid = grid.expand(x.shape[0], *grid.shape)

            transformed_grid = output["transform"][-1](grid)
            

            view = T.broadcasting_grid_sample(x, transformed_grid)

            output["view"] = view # might want this later as well
            split = self.net(view)
            split = self.relu(split)
            reshape = split.view(-1, self.linear_size)
            z_loc = self.loc(reshape)
            z_scale = torch.exp(self.scale(reshape))

            output["z_mu"] = z_loc
            output["z_std"] = z_scale
        else:

            split = self.net(x)
            latents = self.relu(split)
            reshape = latents.view(-1, self.linear_size)
            z_loc = self.loc(reshape)
            z_scale = torch.exp(self.scale(reshape))
            output["z_mu"] = z_loc
            output["z_std"] = z_scale
            params_out = self.layer_for_params(reshape)
            # this will be size (batch, total_num_params)
            params_for_transforms = []
            i = 0
            for num_params in self.parameters_per_transform:
                if num_params == 1:
                    params_for_transforms.append(tuple([params_out[:,i: i+1].squeeze()]))
                    i += 1
                else:
                    params_for_transforms.append(tuple([params_out[:, i: i+1].squeeze(), params_out[:, i+1: i+2].squeeze()]))
                    i += 2
                    
            output["transform_params"] = params_for_transforms
            
            #output["transform"] = self.transformer.transform_from_params(params_for_transforms)
            #TO LEWIS: also not entirely sure how this works above
            #TO LEWIS: do I transfom here or not? i.e do all the broadcasting grid stuff

            output["view"] = x

        return output, split

class Decoder(nn.Module):
    def __init__(self, z_dim=10, outsize=56):
        super().__init__()
        self.z_dim = z_dim
        self.outsize = outsize
        self.linear_size = int((outsize/8)**2)
        self.linear = nn.Linear(z_dim, self.linear_size)
        self.net = nn.Sequential(
            nn.ELU(),
            UpResBloc(1, 32),
            nn.ELU(),
            nn.BatchNorm2d(32),
            ConvBlock(32, bias=False),
            UpResBloc(32, 32),
            nn.ELU(),
            ConvBlock(32, bias=False),
            ConvBlock(32, bias=False),
            UpResBloc(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        z = self.linear(z)
        z = torch.reshape(z, (-1, 1, int(self.outsize/8), int(self.outsize/8)))
        loc_img = self.net(z)
        return loc_img



if __name__ == "__main__":
    x = torch.zeros([10, 1, 80, 80])
    transformers = transformers.TransformerSequence(
        transformers.Translation(networks.EquivariantPosePredictor, 1, 32),
        transformers.Rotation(networks.EquivariantPosePredictor, 1, 32))

    encoder = Encoder(transformers, insize=80, z_dim=10)
    decoder = Decoder(z_dim=10, outsize=80)
    x = encoder(x)
    x = x["z_mu"]
    x = decoder(x)

    
