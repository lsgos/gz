import torch.nn as nn
class PoseVAE(nn.Module):
    def __init__(self, encoder, decoder, z_dim, kwargs_encoder,
                 kwargs_decoder, use_cuda=False):
        super().__init__()
        self.encoder = encoder(**kwargs_encoder)
        self.decoder = decoder(**kwargs_decoder)
        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

def model(
    data,
    transforms=None,
    cond=True,
    decoder=None,
    output_size=40,
    device=torch.device("cpu"),
    **kwargs
):
    decoder = pyro.module("decoder", self.decoder)
    # decoder takes z and std in the transformed coordinate frame
    # and the theta
    # and outputs an upright image
    with pyro.plate(data.shape[0]):
        # prior for z
        z = pyro.sample(
            "z",
            D.Normal(
                torch.zeros(decoder.latent_dim, device=device),
                torch.ones(decoder.latent_dim, device=device),
            ).to_event(1),
        )
        # given a z, the decoder produces an "image"
        # this image must be transformed from the self consistent basis
        # to real world basis
        # first, z and std for the self consistent basis is outputted
        # then it is transfomed
        view = decoder(z)

        pyro.deterministic("canonical_view", view)
        # pyro.deterministic
        # is like pyro.sample but it is deterministic...?

        # all of this is completely independent of the input
        # maybe this is the "prior for the transformation"
        # and hence it looks completely independent of the input
        # but when the model is run again, these variables are replayed
        # with the theta generated by the guide
        # makes sense
        # so the model replays with theta and mu and sigma generated by
        # the guide,
        # taking theta and mu sigma and applying the inverse transform
        # to get the output image.
        grid = coordinates.identity_grid([output_size, output_size], device=device)
        grid = grid.expand(data.shape[0], *grid.shape)

        transform = random_pose_transform(transforms)

        transform_grid = transform(grid)

        # output from decoder is transormed in do a different coordinate system
        
        transformed_view = T.broadcasting_grid_sample(view, transform_grid)
        obs = data if cond else None

        # view from decoder outputs an image 
        pyro.sample("pixels", D.Bernoulli(transformed_view).to_event(3), obs=obs)

    
def guide(data, encoder=None, **kwargs):
    """
    remember the guide is p(z)
    it will sample a z given the x.
    This is parameterised by the weights of the encoder

    """
    encoder = pyro.module("encoder", self.encoder)
    with pyro.plate(data.shape[0]):
        # encoder_out is a dictionary
        # gives transform params, z mu and z std
        # pretty standard guide
        encoder_out = encoder(data)
        
        delta_sample_transformer_params(
            encoder.transformer.transformers, encoder_out["transform_params"]
        )
        # I think this is essentially "sampling" the transform params
        # this will be used when replaying the model to create
        # the image

        z = pyro.sample(
            "z",
            D.Normal(
                encoder_out["z_mu"], torch.exp(encoder_out["z_std"]) + 1e-3
            ).to_event(1),
        )