from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
import numpy as np
from construct_vae import VAE, evaluate, train_log_vae
import torch
import torch.nn.functional as f
import torchvision as tv
import os
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as D
import importlib
from classifier_gz import Classifier
from load_gz_data import Gz2_data, return_data_loader, return_ss_loader
from torch.utils.data import DataLoader
from pyro.infer import SVI, Trace_ELBO
import argparse
from torch.optim import Adam
parser = argparse.ArgumentParser()
csv = "gz2_data/gz_amended.csv"
img = "gz2_data/"

parser.add_argument('--dir_name', required=True)
parser.add_argument('--arch', required=True)
parser.add_argument('--csv_file', metavar='c', type=str, default=csv)
parser.add_argument('--img_file', metavar='i', type=str, default=img)
parser.add_argument('--no_cuda', default=False, action='store_true')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--img_size', default=80, type=int)
parser.add_argument('--lr', default=1.0e-4, type=float)
parser.add_argument('--z_size', default=100, type=int)
parser.add_argument('--crop_size', default=80, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--subset', default=False, action='store_true')
parser.add_argument('--load_checkpoint', default=None)
parser.add_argument('--bar_no_bar', default=False, action='store_true')
parser.add_argument('--dont_detach', default=False, action='store_true')
parser.add_argument('--test_proportion', default=0.1, type=float)
parser.add_argument('--us_proportion', default=0.5, type=float)

args = parser.parse_args()
spec = importlib.util.spec_from_file_location("module.name", args.arch)
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)
Encoder = arch.Encoder
Decoder = arch.Decoder

use_cuda = not args.no_cuda
a01 = "t01_smooth_or_features_a01_smooth_count"
a02 = "t01_smooth_or_features_a02_features_or_disk_count"
a03 = "t01_smooth_or_features_a03_star_or_artifact_count"

if args.bar_no_bar == False:
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
    list_of_ans = [a01, a02, a03]
else:
    a01 = "t03_bar_a06_bar_count"
    a02 = "t03_bar_a07_no_bar_count"
    list_of_ans = [a01, a02]




def evaluate_vae_classifier(vae, vae_loss_fn, classifier, classifier_loss_fn, test_loader, use_cuda=False, transform=False):
    """
    evaluates for all test data
    test data is in batches, all batches in test loader tested
    """
    epoch_loss_vae = 0.
    epoch_loss_classifier = 0.
    total_acc = 0.
    rms = 0.
    for data in test_loader:
        x = data['image']
        y = data['data']
        if transform != False:
            x = transform(x)
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        vae_loss = vae_loss_fn(vae.model, vae.guide, x)
        z_loc, z_scale = vae.encoder(x)
        combined_z = torch.cat((z_loc, z_scale), 1)
        combined_z = combined_z.detach()
        y_out = classifier.forward(combined_z)
        classifier_loss = classifier_loss_fn(y_out, y)
        total_acc += torch.sum(torch.eq(y_out.argmax(dim=1),y.argmax(dim=1)))
        epoch_loss_vae += vae_loss.item()
        epoch_loss_classifier += classifier_loss.item()
        rms += rms_calc(y_out, y)
    normalizer = len(test_loader.dataset)
    total_epoch_loss_vae = epoch_loss_vae / normalizer
    total_epoch_loss_classifier = epoch_loss_classifier / normalizer
    total_epoch_acc = total_acc / normalizer
    rms_epoch = rms / normalizer
    return total_epoch_loss_vae, total_epoch_loss_classifier, total_epoch_acc, rms_epoch


def train_ss_vae_classifier(vae, vae_optim, vae_loss_fn, classifier, classifier_optim, classifier_loss_fn,
                         train_s_loader, train_us_loader, use_cuda=True):
    """
    train vae and classifier for one epoch
    returns loss for one epoch
    in each batch, when the svi takes a step, the optimiser of classifier takes a step
    """
    epoch_loss_vae = 0.
    epoch_loss_classifier = 0.
    total_acc = 0.
    num_steps = 0

    zip_list = zip(train_s_loader, cycle(train_us_loader)) if len(train_s_loader) > len(train_us_loader) else zip(cycle(train_s_loader), train_us_loader)
    for data_sup, data_unsup in zip_list:
        xs = data_sup['image']
        ys = data_sup['data']
        xus = data_unsup['image']
        yus = data_unsup['data']
        if use_cuda:
            xs = xs.cuda()
            ys = ys.cuda()
            xus = xus.cuda()
        classifier_optim.zero_grad()
        vae_optim.zero_grad()
        batch_size = ys.shape[0]
        # supervised step
        
        z_loc, z_scale = vae.encoder(xs)
        combined_z = torch.cat((z_loc, z_scale), 1)
        y_out = classifier.forward(combined_z)
        
        classifier_loss = classifier_loss_fn(y_out, ys)
        vae_loss = vae_loss_fn(vae.model, vae.guide, xs)

        total_loss = vae_loss + classifier_loss
        epoch_loss_vae += vae_loss.item()
        epoch_loss_classifier += classifier_loss.item()
        total_acc += torch.sum(torch.eq(y_out.argmax(dim=1),ys.argmax(dim=1)))
        total_loss.backward()

        vae_optim.step()
        classifier_optim.step()

        # unsupervised step
        
        vae_optim.zero_grad()
        vae_loss = vae_loss_fn(vae.model, vae.guide, xus)
        vae_loss.backward()
        vae_optim.step()
        epoch_loss_vae += vae_loss.item()
        


        num_steps +=1
    total_num_data_stepped_supervised = num_steps * batch_size
    total_num_data_stepped_unsupervised = total_num_data_stepped_supervised * 2

    total_epoch_loss_vae = epoch_loss_vae / total_num_data_stepped_unsupervised
    total_epoch_loss_classifier = epoch_loss_classifier / total_num_data_stepped_supervised
    total_acc_norm = total_acc / total_num_data_stepped_supervised
    return total_epoch_loss_vae, total_epoch_loss_classifier, total_acc_norm

def rms_calc(logits, target):
    """
    total rms for a single batch
    """
    target = target.cpu().numpy()
    probs = f.softmax(logits, dim=1).detach().cpu().numpy()
    total_count = np.sum(target, axis=1)
    probs_target = target / total_count[:, None]
    rms =  np.sqrt((probs - probs_target)**2)
    return np.sum(rms)
    
def train_log_vae_classifier(dir_name, vae, vae_optim, vae_loss_fn, classifier, classifier_optim,
                             classifier_loss_fn, test_s_loader, test_us_loader, train_s_loader, train_us_loader,
                             num_epochs, plot_img_freq=1, num_img_plt=40,
                             checkpoint_freq=20, use_cuda=True, test_freq=1):

    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    writer = SummaryWriter("tb_data_all/" + dir_name)
    if not os.path.exists("checkpoints/" + dir_name):
        os.makedirs("checkpoints/" + dir_name)
    if use_cuda:
        classifier.cuda()
    for epoch in range(num_epochs):
        print("training")
        total_epoch_loss_vae, total_epoch_loss_classifier, total_epoch_acc  = train_ss_vae_classifier(
            vae, vae_optim, vae_loss_fn, classifier,
            classifier_optim, classifier_loss_fn, train_s_loader, train_us_loader, use_cuda=use_cuda)
        print("end train")
        print("[epoch %03d]  average training loss vae: %.4f" % (epoch, total_epoch_loss_vae))
        print("[epoch %03d]  average training loss classifier: %.4f" % (epoch, total_epoch_loss_classifier))
        print("[epoch %03d]  average training accuracy: %.4f" % (epoch, total_epoch_acc))
        
        if epoch % test_freq == 0:
            # report test diagnostics
            print("evaluating")
            total_epoch_loss_test_vae, total_epoch_loss_test_classifier, accuracy, rms = evaluate_vae_classifier(
                vae, vae_loss_fn, classifier, classifier_loss_fn, test_s_loader,
                use_cuda=use_cuda)
            print("[epoch %03d] average test loss vae: %.4f" % (epoch, total_epoch_loss_test_vae))
            print("[epoch %03d] average test loss classifier: %.4f" % (epoch, total_epoch_loss_test_classifier))
            print("[epoch %03d] average accuracy: %.4f" % (epoch, accuracy))
            print("evaluate end")
            writer.add_scalar('Train loss vae', total_epoch_loss_vae, epoch)
            writer.add_scalar('Train loss classifier', total_epoch_loss_classifier, epoch)
            writer.add_scalar('Train accuracy', total_epoch_acc, epoch)
            writer.add_scalar('Test loss vae', total_epoch_loss_test_vae, epoch)
            writer.add_scalar('Test loss classifier', total_epoch_loss_test_classifier, epoch)
            writer.add_scalar('Test accuracy', accuracy, epoch)
            writer.add_scalar('rms normalised', rms, epoch)
            
        if epoch % plot_img_freq == 0:
            
            image_in = next(iter(test_us_loader))['image'][0:num_img_plt]
            images_out = vae.sample_img(image_in, use_cuda=use_cuda)
            img_grid_in = tv.utils.make_grid(image_in)
            img_grid = tv.utils.make_grid(images_out)
            writer.add_image('images in, from epoch' + str(epoch), img_grid_in)
            writer.add_image(str(num_params) + ' images out, from epoch'+ str(epoch), img_grid)

        if epoch % checkpoint_freq == 0:

            torch.save(vae.encoder.state_dict(), "checkpoints/" + dir_name + "/encoder.checkpoint")
            torch.save(vae.decoder.state_dict(),  "checkpoints/" + dir_name +  "/decoder.checkpoint")
            torch.save(classifier.state_dict(),  "checkpoints/" + dir_name +  "/classfier.checkpoint")
            
        writer.close()

data = Gz2_data(csv_dir=args.csv_file,
                image_dir=args.img_file,
                list_of_interest=list_of_ans,
                crop=args.img_size,
                resize=args.crop_size)


encoder_args = {'insize':args.img_size, 'z_dim':args.z_size}
decoder_args = {'z_dim':args.z_size, 'outsize':args.img_size}

if args.subset is True:
    test_s_loader, test_us_loader, train_s_loader, train_us_loader = return_ss_loader(
        data, args.test_proportion, args.us_proportion, batch_size=args.batch_size, shuffle=True, subset=True)
else:
    test_s_loader, test_us_loader, train_s_loader, train_us_loader  = return_ss_loader(
        data, args.test_proportion, args.us_proportion, batch_size=args.batch_size, shuffle=True, subset=False)
print("total data:",  len(data))
print("num data points in test_s_loader:", len(test_s_loader.dataset))
print("num data points in test_us_loader:", len(test_us_loader.dataset))
print("num data points in train_s_loader:", len(train_s_loader.dataset))
print("num data points in train_us_loader:", len(train_us_loader.dataset))
print("train and log")

vae = VAE(Encoder, Decoder, args.z_size, encoder_args, decoder_args, use_cuda=use_cuda)
if args.load_checkpoint != None:
    vae.encoder.load_state_dict(torch.load("checkpoints/" + args.load_checkpoint + "/encoder.checkpoint"))
    vae.decoder.load_state_dict(torch.load("checkpoints/" + args.load_checkpoint + "/decoder.checkpoint"))
vae_optim = Adam(vae.parameters(), lr= args.lr, betas= (0.90, 0.999))

classifier = Classifier(in_dim=args.z_size*2)

classifier_optim = Adam(classifier.parameters(),args.lr /10 , betas=(0.90, 0.999))
# or optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)?

def multinomial_loss(logits, values):
    return torch.sum(-1 *D.Multinomial(1, logits=logits).log_prob(values.float()))

classifier_loss = multinomial_loss

train_log_vae_classifier(args.dir_name, vae, vae_optim, Trace_ELBO().differentiable_loss,
                         classifier, classifier_optim,
                         classifier_loss, test_s_loader, test_us_loader, train_s_loader, train_us_loader,
                         args.num_epochs, use_cuda=use_cuda, )