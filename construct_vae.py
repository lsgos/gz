import os

import torch.nn as nn
import pyro

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
import pyro.distributions as D

from galaxy_gen.backward_models import delta_sample_transformer_params
from galaxy_gen.etn import transforms as T
from galaxy_gen.etn import transformers, networks
from galaxy_gen.etn import coordinates
from galaxy_gen.forward_models import random_pose_transform
from utils.load_gz_data import Gz2_data, return_data_loader


class PoseVAE(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 z_dim, kwargs_encoder,
                 kwargs_decoder, transforms,
                 use_cuda,
                 pixel_likelihood = 'bernoulli',
                 ):
        super().__init__()
        self.encoder = encoder(**kwargs_encoder)
        self.decoder = decoder(**kwargs_decoder)
        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.transforms = transforms
        self.pixel_likelihood = pixel_likelihood

    def model(self,
              data):

        output_size = self.encoder.insize
        decoder = pyro.module("decoder", self.decoder)
        # decoder takes z and std in the transformed coordinate frame
        # and the theta
        # and outputs an upright image
        with pyro.plate(data.shape[0]):
            # prior for z
            z = pyro.sample(
                "z",
                D.Normal(
                    torch.zeros(decoder.z_dim, device=data.device),
                    torch.ones(decoder.z_dim, device=data.device),
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
            grid = coordinates.identity_grid(
                [output_size, output_size], device=data.device)
            grid = grid.expand(data.shape[0], *grid.shape)
            transform = random_pose_transform(self.transforms)

            transform_grid = transform(grid)

            # output from decoder is transormed in do a different coordinate system

            transformed_view = T.broadcasting_grid_sample(view, transform_grid)

            # view from decoder outputs an image
            if self.pixel_likelihood == 'bernoulli':
                pyro.sample(
                    "pixels", D.Bernoulli(transformed_view).to_event(3), obs=data)
            elif self.pixel_likelihood == 'laplace':
                pyro.sample(
                      "pixels", D.Laplace(transformed_view, 0.5).to_event(3), obs=data)
            else:
                raise NotImplementedError(f"Unimplemented pixel likelihood {self.pixel_likelihood}")

    def guide(self, data):
        """
        remember the guide is p(z)
        it will sample a z given the x.
        This is parameterised by the weights of the encoder
        Guide samples mu and std in the transformed space.
        The decoder outputs the "image" in the transfomed coordinate system
        """
        encoder = pyro.module("encoder", self.encoder)
        with pyro.plate(data.shape[0]):
            # encoder_out is a dictionary
            # gives transform params, z mu and z std
            # pretty standard guide
            encoder_out, split = encoder(data)

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

    def sample_img(self, x, use_cuda, encoder=False, decoder=False):
        # encode image x
        if use_cuda is True:
            x = x.cuda()
        batch_shape = x.shape[0]
        img_shape = x.shape[-1]
        if encoder is False:
            out, split = self.encoder(x)
        else:
            out, split = encoder(x)
        # sample in latent space
        z = out["z_mu"]
        # decode the image (note we don't sample in image space)
        if decoder is False:
            loc_img = self.decoder(z)
        else:
            loc_img = decoder(z)
        return loc_img.reshape([batch_shape, 1, img_shape, img_shape])


def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    transforms = T.TransformSequence(T.Translation(), T.Rotation())
    for x in train_loader:
        x = x['image']
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x, transforms)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    transforms = T.TransformSequence(T.Translation(), T.Rotation())
    # compute the loss over the entire test set
    for x in test_loader:
        # if on GPU put mini-batch into CUDA memory
        x = x['image']
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x, transforms)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def train_log(dir_name, vae, svi, train_loader, test_loader,
              num_epochs, plot_img_freq=20, num_img_plt=10,
              checkpoint_freq=20, use_cuda=True, test_freq=1):

    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

    writer = SummaryWriter("tb_data_all/" + dir_name)
    if not os.path.exists("checkpoints/" + dir_name):
        os.makedirs("checkpoints/" + dir_name)
    for epoch in range(num_epochs):
        print("training")
        total_epoch_loss_train = train(svi, train_loader, use_cuda=use_cuda)
        print("end train")
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
        if epoch % test_freq == 0:
            # report test diagnostics
            print("evaluating")
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
            print("evaluate end")
            writer.add_scalar('Train loss', total_epoch_loss_train, epoch)
            writer.add_scalar('Test loss', total_epoch_loss_test, epoch)
            print(epoch)
        if epoch % plot_img_freq == 0:
            image_in = next(iter(train_loader))['image'][0:num_img_plt]
            images_out = vae.sample_img(image_in, use_cuda=use_cuda)
            img_grid_in = tv.utils.make_grid(image_in)
            img_grid = tv.utils.make_grid(images_out)
            writer.add_image('images in, from epoch' + str(epoch), img_grid_in)
            writer.add_image(str(num_params) + ' images out, from epoch'+ str(epoch), img_grid)

        if epoch % checkpoint_freq == 0:

            torch.save(
                vae.encoder.state_dict(), "checkpoints/" + dir_name + "/encoder.checkpoint")
            torch.save(
                vae.decoder.state_dict(),  "checkpoints/" + dir_name +  "/decoder.checkpoint")

        writer.close()


if __name__ == "__main__":
    csv = "gz2_data/gz_amended.csv"
    img = "gz2_data/"
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
    list_of_ans = [a01, a02, a03]

    data = Gz2_data(csv_dir=csv,
                    image_dir=img,
                    list_of_interest=list_of_ans,
                    crop=80,
                    resize=80)

    trans = transformers.TransformerSequence(
        transformers.Translation(networks.EquivariantPosePredictor, 1, 32),
        transformers.Rotation(networks.EquivariantPosePredictor, 1, 32))
    encoder_args = {'insize': 80, 'z_dim': 100, 'transformer': trans}
    decoder_args = {'z_dim': 100, 'outsize': 80}

    test_proportion = 0.1
    train_loader, test_loader  = return_data_loader(data, test_proportion, batch_size=2, shuffle=True)
    optimizer = Adam({"lr": 1e-4})
    pvae = PoseVAE(Encoder, Decoder, 10, encoder_args, decoder_args, use_cuda=False)
    svi = SVI(pvae.model, pvae.guide, optimizer, loss=Trace_ELBO())

    train_log("test/", pvae, svi, train_loader, test_loader, 1, use_cuda=False)
