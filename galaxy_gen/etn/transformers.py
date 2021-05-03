"""
Transformer modules.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .coordinates import *
from . import transforms as T

    
    
class Transformer(nn.Module):
    def __init__(self, predictor_cls, in_channels, nf, transform=T.Transform(),
                 coords=identity_grid, 
                 ulim=None, vlim=None, 
                 return_u=True, return_v=True,
                 periodic_u=False, periodic_v=False,
                 rescale=True,
                 **kwargs):
        """Transformer module base class. 
        
        Args:
            predictor_cls: Callable, instantiates an nn.Module instance for predicting 
                pose parameters.
            in_channels: int, Size of channel dimension of input tensor.
            nf: int, Number of filters for instantiating pose predictor.
            coords: Callable, coordinate transformation
            ulim: (float, float), limits of u coordinate
            vlim: (float, float), limits of v coordinate
            return_u: bool, whether to return a prediction for the u coordinate.
            return_v: bool, whether to return a prediction for the v coordinate.
            periodic_u: bool, whether the u coordinate is periodic.
            periodic_v: bool, whether the v coordinate is periodic.
            rescale: bool, whether to scale the predicted u and v by half the range
                of ulim and vlim. Useful for pose predictors that return values in [-1, 1].
        """
        super().__init__()
        self.coords = coords
        self.ulim = ulim
        self.vlim = vlim
        self.return_u = return_u
        self.return_v = return_v
        self.periodic_u = periodic_u
        self.periodic_v = periodic_v or transform.periodic_v     
        self.rescale = rescale
        self.transform = transform
        
        self.num_outputs = 2 if (return_u and return_v) else 1
        self.predictor = predictor_cls(
            in_channels=in_channels, 
            nf=nf,
            periodic_u=periodic_u,
            periodic_v=periodic_v,
            return_u=return_u,
            return_v=return_v,
            num_outputs=self.num_outputs,
            **kwargs)

    def transform_from_params(self, params):
        """Returns a transformation function from the given parameters"""
        return self.transform.transform_from_params(*params)
    
    def inverse_transform_from_params(self, params):
        return self.transform.inverse_transform_from_params(*params)

    def forward(self, x, transform=None, grid_size=None, padding_mode='zeros'):
        """ 
        Args:
            x: torch.Tensor, Input tensor in NCHW format.
            transform: GridTransform, a grid-to-grid transformation that takes an input a
                [H, W, 2]-shaped tensor of x-y coordinates of grid points and outputs
                a tensor of the same shape. This implicitly represents the transformation
                applied to the input image before calling the current Transformer module.
            grid_size: (int, int), Shape of the grid used to predict pose parameters.
                Uses the height and width of the input `x` by default.
            padding_mode: str, option from `torch.nn.functional.grid_sample`. 
                Valid values are "zeros", "border", and "reflection".
                
        Return:
            dict, {'transform': predicted transformation, 
                   'params': predicted transformation parameters, 
                   'maps': heatmaps used to predict transformation parameters}
            
        """
        if grid_size is None:
            grid_size = x.shape[-2:]
        
        grid = self.coords(grid_size, ulim=self.ulim, vlim=self.vlim, device=x.device)
        grid = grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        if transform is not None:
            grid = transform(grid)
        
        x_tf = T.broadcasting_grid_sample(x, grid, padding_mode=padding_mode)        
        
        coords, heatmaps = self.predictor(x_tf)
        
        if self.rescale:
            urad = (self.ulim[1] - self.ulim[0]) / 2.
            vrad = (self.vlim[1] - self.vlim[0]) / 2.
        else:
            urad = 1.
            vrad = 1.
            
        if self.return_u and self.return_v:
            u, v = coords
            u = u.mul(urad)
            v = v.mul(vrad)
            params = (u, v)
        elif self.return_u:
            u, v = coords, None
            u = u.mul(urad)
            params = (u,)
        else:
            u, v = None, coords
            v = v.mul(vrad)
            params = (v,)           
        
        new_transform = self.transform_from_params(params)
        if transform is not None:
            new_transform = transform.compose(new_transform)
        
        return {
            'transform': new_transform,
            'params': params,
            'maps': heatmaps,
        }
    
    
class TransformerSequence(nn.Module):
    def __init__(self, *transformers):
        """A container class representing a sequence of Transformer modules to be applied iteratively.
        
        Args:
            transformers: a sequence of Transformer modules.
        """
        super().__init__()
        self.transformers = nn.ModuleList(transformers)

    def forward(self, x, transform=None, grid_size=None, padding_mode='zeros'):            
        params = []
        heatmaps = []
        transforms = []

        if len(self.transformers) == 0:
            # handle an edge case. This let's us turn of the transformer without 
            # modifying outlying code so much.
            transforms.append(T.ProjectiveGridTransform(torch.eye(3, device=torch.device('cuda')))) 
        # fold over projective modules
        for i, tf in enumerate(self.transformers):
            out_dict = tf(x, transform, grid_size=grid_size, padding_mode=padding_mode)
            transform = out_dict['transform']
            transforms.append(transform)
            params.append(out_dict['params'])
            heatmaps.append(out_dict['maps'])
            
        return {
            'transform': transforms,
            'params': params,
            'maps': heatmaps,
        }


class TransformerParallel(nn.Module):
    def __init__(self, *transformers):
        """A container class representing a sequence of Transformer modules to be applied in parallel.
        
        Args:
            transformers: a sequence of Transformer modules.
        """
        super().__init__()
        self.transformers = nn.ModuleList(transformers)

    def forward(self, x, transform=None, grid_size=None, padding_mode='zeros'):            
        params = []
        heatmaps = []
        par_transforms = []
        
        # fold over projective modules
        for i, tf in enumerate(self.transformers):
            out_dict = tf(x, transform=transform, grid_size=grid_size, padding_mode=padding_mode)
            par_transforms.append(out_dict['transform'])
            params.append(out_dict['params'])
            heatmaps.append(out_dict['maps'])
        
        transforms = [par_transforms[0]]
        for tf in par_transforms[1:]:
            transforms.append(transforms[-1].compose(tf))
            
        return {
            'transform': transforms,
            'params': params,
            'maps': heatmaps,
        }


class Translation(Transformer):
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=identity_grid,
                 ulim=(-1, 1), 
                 vlim=(-1, 1), 
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.Translation(),
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         **kwargs)             
        
class Rotation(Transformer):    
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=polar_grid,
                 ulim=(0., np.sqrt(2.)),
                 vlim=(-np.pi, np.pi),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.Rotation(),
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_u=False,
                         periodic_v=True,
                         **kwargs)       
        
class Scale(Transformer):
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=logpolar_grid,
                 ulim=(-np.log(10.), np.log(2.)/2.),
                 vlim=(-np.pi, np.pi),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls, 
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.Scale(),
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_v=False,
                         periodic_v=True,
                         **kwargs)               
        
class RotationScale(Transformer):    
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=logpolar_grid,
                 ulim=(-np.log(10.), np.log(2.)/2.),
                 vlim=(-np.pi, np.pi),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls, 
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.RotationScale(),
                         coords=coords,
                         ulim=ulim,                         
                         vlim=vlim,
                         periodic_v=True,
                         **kwargs)
        
class ShearX(Transformer):
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=shearx_grid,
                 ulim=(-1, 1),
                 vlim=(-5, 5),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls, 
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.ShearX(),
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_u=False,
                         **kwargs)
        
class ShearY(Transformer):
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=sheary_grid,
                 ulim=(-1, 1),
                 vlim=(-5, 5),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls, 
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.ShearY(),
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_u=False,
                         **kwargs)
    
class ScaleX(Transformer):    
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=scalex_grid,
                 ulim=(-np.log(10.), 0),                 
                 vlim=(-1, 1),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.ScaleX(),
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_v=False,
                         **kwargs)       
    
class ScaleY(Transformer):    
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=scalex_grid,
                 ulim=(-np.log(10.), 0),                 
                 vlim=(-1, 1),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.ScaleY(),
                         coords=coords,
                         ulim=ulim,                         
                         vlim=vlim,
                         return_v=False,
                         **kwargs)
    
class HyperbolicRotation(Transformer):
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=hyperbolic_grid,
                 ulim=(-np.sqrt(0.5), np.sqrt(0.5)),
                 vlim=(-np.log(6.), np.log(6.)),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls,
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.HyperbolicRotation(),
                         coords=coords,
                         ulim=ulim,
                         vlim=vlim,
                         return_u=False,
                         **kwargs)
        
class PerspectiveX(Transformer):
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=perspectivex_grid,
                 ulim=(1, 7),                 
                 vlim=(-0.99*np.pi/2, 0.99*np.pi/2),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls, 
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.PerspectiveX(),
                         coords=coords,
                         ulim=ulim,                        
                         vlim=vlim,
                         return_v=False,
                         **kwargs)    
               
class PerspectiveY(Transformer):
    def __init__(self, predictor_cls, in_channels, nf,
                 coords=perspectivey_grid,
                 ulim=(1, 7),                 
                 vlim=(-0.99*np.pi/2, 0.99*np.pi/2),
                 **kwargs):
        super().__init__(predictor_cls=predictor_cls, 
                         in_channels=in_channels,
                         nf=nf,
                         transform=T.PerspectiveY(),
                         coords=coords,
                         ulim=ulim,                        
                         vlim=vlim,
                         return_v=False,
                         **kwargs)
               
