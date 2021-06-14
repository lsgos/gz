import math
from collections import namedtuple
import numpy as np
import importlib
import os
# from tqdm.auto import tqdm
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms as tvt
from kornia.augmentation import RandomRotation
from construct_vae import PoseVAE
from sacred import Experiment
from pyro.infer import Trace_ELBO
import pyro
from utils.load_gz_data import (
    Gz2_data,
)  # , return_data_loader, return_subset, return_ss_loader


from torch import distributions as D

from batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
    repeated_mnist,
)

from trainer import get_transformations



ex = Experiment()

local_csv_loc = "~/diss/gz2_data/gz_amended.csv"
local_img_loc = "~/diss/gz2_data/"
local_fashion_loc = "~/diss/"
run_local = False
fashion_loc = local_fashion_loc if run_local else "/scratch-ssd/oatml/data/"

@ex.config
def config():
    dir_name = "fashion_bar"
    cuda = True
    num_epochs = 200
    semi_supervised = True
    split_early = False
    subset_proportion = None
    csv_file = local_csv_loc if run_local else "/scratch/gz2/gz2_classifications_and_subjects.csv"
    img_file = local_img_loc if run_local else "/scratch/gz2"
    load_checkpoint = False
    lr = 1.0e-4
    arch_classifier = "neural_networks/classifier_fc.py"
    arch_vae = "neural_networks/encoder_decoder.py"
    test_proportion = 0.1
    z_size = 128
    bar_no_bar = False
    batch_size = 64
    crop_size = 128
    use_subset = False
    split_early = False
    use_pose_encoder = True # should also add an option to use a pretrained model
    classify_from_z = False
    pretrain_epochs = 100
    transform_spec = ["Translation", "Rotation"]
    dataset = "FashionMNIST"
    img_size = 32 if dataset == "FashionMNIST" else 128
    acquisition = "BALD"
    pixel_likelihood= 'laplace'
    spatial_vae = False
    data_aug = True
    spatial_transformer = False
    train_vae_only = False
    vae_checkpoint = None


class BayesianCNN(consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1600, 128)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))

        input = torch.flatten(input, -3, -1)

        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input


class MikeCNN(
    consistent_mc_dropout.BayesianModule
):  # the exact architecture used in Walmsley, Smith et. al.
    def __init__(self, nc):
        super().__init__()

        def make_conv(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, kernel_size=3, padding=1),
                nn.ReLU(),
                consistent_mc_dropout.ConsistentMCDropout2d(p=0.0),
            )

        self.body = nn.Sequential(
            make_conv(1, 32),  # 128
            make_conv(32, 32),
            nn.MaxPool2d(2),  # 64
            make_conv(32, 32),
            make_conv(32, 32),
            nn.MaxPool2d(2),  # 32
            make_conv(32, 16),
            nn.MaxPool2d(2),  # 16
            make_conv(16, 16),
            nn.MaxPool2d(2),  # 8
            nn.Flatten(-3, -1),
            nn.Linear(16 * 8 * 8, 128),
            consistent_mc_dropout.ConsistentMCDropout(),
            nn.Linear(128, nc),
        )

    def mc_forward_impl(self, x):
        return self.body(x)

class FCNN(consistent_mc_dropout.BayesianModule):
    def __init__(self, ni, nc, n_hid = 256):
        super().__init__()
        self.body = nn.Sequential(
                nn.Linear(ni,n_hid),
                consistent_mc_dropout.ConsistentMCDropout(),
                nn.ReLU(),
                nn.Linear(n_hid, n_hid),
                consistent_mc_dropout.ConsistentMCDropout(),
                nn.ReLU(),
                nn.Linear(n_hid, nc)
        )
    def mc_forward_impl(self, x):
        return self.body(x)

@ex.capture
def get_datasets(dataset, data_aug):
    datasets = {"FashionMNIST": get_fashionMNIST, "gz": get_gz_data}
    f = datasets.get(dataset)
    if f is not None:
        return f()
    else:
        raise NotImplementedError(
            f"Unk;/nown dataset {dataset}, avaliable options are {set(datasets.keys())}"
        )


class TensorDatasetWithTransform(torch.utils.data.TensorDataset):
    """
    Variant of TensorDataset which allows applying a transform to the first tensor while indexing.
    """
    def __init__(self, *tensors, transform=lambda x: x):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, i):
        x, *rest = super().__getitem__(i)
        return self.transform(x), *rest

@ex.capture
def get_fashionMNIST(dataset, data_aug):
    train_dataset = datasets.FashionMNIST(
        fashion_loc, download=True, train=True, transform=tvt.ToTensor()
    )
    test_dataset = datasets.FashionMNIST(
        fashion_loc, download=True, train=False, transform=tvt.ToTensor()
    )
    # want to create a rotated dataset but not by resampling rotations randomly, as this kind of screws up the active learning argument.
    transform = RandomRotation(180)
    td = train_dataset.data[:, None, ...].float() / 255
    vd = test_dataset.data[:, None, ...].float() / 255

    tt = train_dataset.targets
    vt = test_dataset.targets

    # apply transforms and pad
    td = transform(td)
    vd = transform(vd)

    td = F.pad(td, (2, 2, 2, 2))
    vd = F.pad(vd, (2, 2, 2, 2))
    if data_aug:
        tds = TensorDatasetWithTransform(td, tt, transform=tvt.Compose([tvt.ToPILImage(), tvt.RandomRotation(180), tvt.ToTensor()]))
        vds = TensorDatasetWithTransform(vd, vt, transform=tvt.Compose([tvt.ToPILImage(), tvt.RandomRotation(180), tvt.ToTensor()]))
    else:
        tds = torch.utils.data.TensorDataset(td, tt)
        vds = torch.utils.data.TensorDataset(vd, vt)

    tds.data = tds.tensors[0]
    tds.targets = tds.tensors[1]

    vds.data = vds.tensors[0]
    vds.targets = vds.tensors[1]
    return tds, vds


@ex.capture
def get_model(
    arch_classifier, arch_vae, transform_spec, split_early, z_size, img_size, cuda,
        classify_from_z, pixel_likelihood, spatial_vae, spatial_transformer
):
    ### loading classifier network
    spec = importlib.util.spec_from_file_location("module.name", arch_classifier)
    class_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(class_arch)
    Classifier = class_arch.Classifier

    ### setting up encoder, decoder and vae
    spec = importlib.util.spec_from_file_location("module.name", arch_vae)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    Encoder = arch.Encoder
    Decoder = arch.Decoder
    transforms, transformer = get_transformations(transform_spec, spatial_transformer=spatial_transformer)
    encoder_args = {"transformer": transformer, "insize": img_size, "z_dim": z_size, "spatial_vae": spatial_vae}
    decoder_args = {"z_dim": z_size, "outsize": img_size}
    vae = PoseVAE(
        Encoder, Decoder, z_size, encoder_args, decoder_args, transforms, use_cuda=cuda, pixel_likelihood=pixel_likelihood
    )

    if split_early:
        classifier = Classifier(in_dim=vae.encoder.linear_size)
    else:
        classifier = Classifier(in_dim=z_size)

    return vae, classifier


@ex.capture
def get_gz_data(csv_file, img_file, bar_no_bar, img_size, crop_size, test_proportion, data_aug):
    ans = {
        False: [
            "t01_smooth_or_features_a01_smooth_count",
            "t01_smooth_or_features_a02_features_or_disk_count",
            # "t01_smooth_or_features_a03_star_or_artifact_count", # simply ignore this answer - Mike did
        ],
        True: [
            "t03_bar_a06_bar_count",
            "t03_bar_a07_no_bar_count",
        ],
    }

    data_train = Gz2_data(
        csv_dir=csv_file,
        image_dir=img_file,
        list_of_interest=ans[bar_no_bar],
        crop=img_size,
        resize=crop_size,
        data_aug=data_aug
    )
    data_test = Gz2_data(
        csv_dir=csv_file,
        image_dir=img_file,
        list_of_interest=ans[bar_no_bar],
        crop=img_size,
        resize=crop_size,
        data_aug=False
    )


    len_data = len(data_train)
    num_tests = int(len_data * test_proportion)
    test_indices = list(i for i in range(0, num_tests))
    train_indices = list(i for i in range(num_tests, len_data))
    test_set = torch.utils.data.Subset(data_test, test_indices)
    train_set = torch.utils.data.Subset(data_train, train_indices)
    return train_set, test_set

@ex.capture
def get_classification_model(dataset, bar_no_bar, z_size, classify_from_z):
    if dataset == "FashionMNIST" and not classify_from_z:
        return BayesianCNN()
    if dataset == "FashionMNIST" and classify_from_z:
        raise NotImplementedError
    else:
        if classify_from_z:
            return FCNN(z_size, 2)
        return MikeCNN(2)




def multinomial_loss(logits, observations, reduction="mean"):
    """
    the nll of a multinomial distirbution parameterised by logits.
    """
    p = D.Multinomial(logits=logits)
    if reduction=="mean":
        return -p.log_prob(observations).mean()
    elif reduction == "sum":
        return -p.log_prob(observations).sum()
    raise NotImplementedError(f"Unknown reduction {reduction}")

@ex.capture
def get_classification_loss(dataset):
    if dataset == "gz":
        return multinomial_loss
    else:
        return F.nll_loss

@ex.capture
def preprocess_batch(data, dataset, use_pose_encoder, classify_from_z, vae=None):
    """
    depending on the configuration, either just normalise the data, or pass it through a VAE or similar
    """

    if use_pose_encoder: # could replace this with a configurable function
        enc_output, _ = vae.encoder(data)  # TODO visualise
        if classify_from_z:
            data = enc_output["z_mu"] # technically ought to average here but this will do for now
        else:
            data = enc_output["view"]  # learned transform of the data.
    else:
        if dataset=="gz":
            data = data - 0.222 # lol this complicates switching datasets severely
            data = data / 0.156
    return data


@ex.automain
def main(use_pose_encoder, pretrain_epochs, dataset, lr, bar_no_bar, acquisition, train_vae_only, dir_name, vae_checkpoint, use_subset, _seed, cuda, data_aug, _run):
    os.system('nvidia-smi -q | grep UUID')
    pyro.set_rng_seed(_seed)
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    print(_seed)
    train_dataset_aug, test_dataset = get_datasets(dataset, data_aug)
    train_dataset_no_aug, test_dataset = get_datasets(dataset, False)
    assert acquisition in {"BALD", "random"}, f"Unknown acquisition {acquisition}"


    # sanity for initial training
    if use_subset:
        # purely for iteration purposes, train on a manageable subset of the data
        train_dataset_aug = torch.utils.data.Subset(train_dataset_aug, torch.arange(60000))

    num_initial_samples = 512 if bar_no_bar else 256
    num_classes = 10

    # TODO going to have to change this for galaxy zoo in all likelihood but it will do for now
    if dataset == "gz":
        # in this case getting initial samples is slightly complicated by the fact that galaxy zoo does not have
        # strict labels.
        # Compromise by balancing as though the arg-max of the votes is a label, which should be a pretty good proxy.
        labels = [x['data'].argmax(-1) for x in train_dataset_no_aug]
        num_classes = 2 #  if bar_no_bar else 3
        initial_samples = active_learning.get_balanced_sample_indices(
            labels,
            num_classes=num_classes,
            n_per_digit=num_initial_samples / num_classes
        )
    else:
        initial_samples = active_learning.get_balanced_sample_indices(
            repeated_mnist.get_targets(train_dataset_no_aug),
            num_classes=num_classes,
            n_per_digit=num_initial_samples / num_classes,
        )

    max_training_samples = 5000
    acquisition_batch_size = 256 if bar_no_bar else 128
    num_inference_samples = 5
    num_test_inference_samples = 25
    # num_samples = 100000

    test_batch_size = 32
    batch_size = 64
    scoring_batch_size = 32
    training_iterations = 4096 * 16

    device = "cuda" if cuda else "cpu"
    print("device: {}".format(device))
    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs
    )

    active_learning_data = active_learning.ActiveLearningData(train_dataset_no_aug)
    active_learning_data_aug = active_learning.ActiveLearningData(train_dataset_aug)

    # Split off the initial samples first.
    active_learning_data.acquire(initial_samples)
    active_learning_data_aug.acquire(initial_samples)

    train_loader_aug = torch.utils.data.DataLoader(
        active_learning_data_aug.training_dataset,
        sampler=active_learning.RandomFixedLengthSampler(
            active_learning_data_aug.training_dataset, training_iterations
        ),
        batch_size=batch_size,
        **kwargs,
    )

    pool_loader_aug = torch.utils.data.DataLoader(
        active_learning_data_aug.pool_dataset,
        batch_size=scoring_batch_size,
        shuffle=False,
        **kwargs,
    )

    # Run experiment
    test_accs = []
    test_loss = []
    test_rmse = []
    added_indices = []

    # pbar = tqdm(
    #     initial=len(active_learning_data.training_dataset),
    #     total=max_training_samples,
    #     desc="Training Set Size",
    # )

    vae, _ = get_model()
    vae_opt = torch.optim.Adam(vae.parameters(), lr=lr)
    vae_loader = torch.utils.data.DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True)
    print("starting pretraining")
    if use_pose_encoder:
        if vae_checkpoint is not None:
            print("loaded checkpoints from: ", vae_checkpoint )
            vae.encoder.load_state_dict(
                torch.load(vae_checkpoint + "/encoder.checkpoint"))
            vae.decoder.load_state_dict(
                torch.load(vae_checkpoint + "/decoder.checkpoint"))
        else:
            for e in range(pretrain_epochs):
                lb = []
                for batch in vae_loader:
                    if isinstance(batch, dict):
                        x = batch['image']
                    else:
                        x = batch[0]
                    x = x.to(device)
                    vae_opt.zero_grad()

                    loss = (
                        Trace_ELBO().differentiable_loss(vae.model, vae.guide, x)
                        / batch_size
                    )
                    loss.backward()
                    vae_opt.step()
                    lb.append(loss.item())
                print("pretrain epoch", e, "average loss", np.mean(lb))
            print("done pretraining")
    if train_vae_only:
        print("trying to save checkpoints")
        torch.save(vae.encoder.state_dict(), "checkpoints/" + dir_name + "/encoder.checkpoint")
        torch.save(vae.decoder.state_dict(),  "checkpoints/" + dir_name +  "/decoder.checkpoint")
        print("checkpoints saved to {}".format("checkpoints/" + dir_name))
        return

    # todo want a switch on BayesianCNN / PoseVAE here.
    first = True
    model = get_classification_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = get_classification_loss()
    model.to(device)
    model.train()
    while True:

        # Train
        first = False
        n_repeats = 3 if first else 1
        lb = []
        model.train()
        for _ in range(n_repeats):
            for batch in train_loader_aug:

                if isinstance(batch, dict):
                    data = batch['image']
                    target = batch['data']

                else:
                    data, target = batch

                data = data.to(device=device)
                target = target.to(device=device)
                data = preprocess_batch(data, vae=vae)

                optimizer.zero_grad()

                prediction = model(data, 1).squeeze(1)
                # loss = F.nll_loss(prediction, target)
                if prediction.shape[0] != target.shape[0]:
                    breakpoint()
                loss = loss_fn(prediction, target)
                # TODO juts sanity
                # fake_target = target.float() / target.sum(-1, keepdim=True)
                # loss = ((prediction - fake_target) **2).mean()

                lb.append(loss.item())
                # also record running training loss as a sanity check
                if len(lb) % 100 == 0:
                    _run.log_scalar('train.loss_running', np.mean(lb[-100:]))
                loss.backward()

                optimizer.step()
        # print('train loss:',  np.mean(lb))
        _run.log_scalar('train.loss', np.mean(lb))
        # Test
        loss = 0
        correct = 0
        mse = 0
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    data = batch['image']
                    target = batch['data']

                else:
                    data, target = batch

                data = data.to(device=device)
                target = target.to(device=device)

                data = preprocess_batch(data, vae=vae)

                model_out = model(data, num_test_inference_samples)
                log_preds = F.log_softmax(model(data, num_test_inference_samples), -1)
                prediction = torch.logsumexp(
                    log_preds, dim=1
                ) - math.log(num_test_inference_samples)
                # loss += F.nll_loss(prediction, target, reduction="sum")
                if dataset == 'gz':
                    class_pred = prediction.max(1)[1]
                    if len(target.shape) > 1:
                        class_true = target.argmax(-1)
                    else:
                        class_true = target
                    correct += class_pred.eq(class_true.view_as(class_pred)).sum().item()

                    # calculate RMSE between predictions and target.
                    emp_probs = target.float() / target.float().sum(-1, keepdim=True)
                    pred_prob = torch.exp(prediction)
                    loss += loss_fn(prediction, target) * target.shape[0]  # loss_fn takes mean, but we want the sum.
                    assert emp_probs.shape[-1] == 2
                    ep = emp_probs[..., 0]
                    pp = pred_prob[..., 0]
                    mse += ((ep - pp) **2).sum().item()
                elif dataset == 'FashionMNIST':
                    class_pred = prediction.max(1)[1]
                    correct += (class_pred == target).float().sum(-1).item()
                    loss += loss_fn(prediction, target).item() * target.shape[0]
                    mse += 0

        # double check scaling here.
        loss /= len(test_loader.dataset)
        test_loss.append(loss)

        percentage_correct = 100.0 * correct / len(test_loader.dataset)
        test_accs.append(percentage_correct)

        mse = mse / len(test_loader.dataset)
        rmse = math.sqrt(mse)
        test_rmse.append(rmse)


        assert type(loss) == float # double check these aren't torch tensors - this screws up storage
        assert type(rmse) == float
        _run.log_scalar("test.average_loss", loss)
        _run.log_scalar("test.average_accuracy", percentage_correct)
        _run.log_scalar("test.rmse", rmse) ## todo double check how these are calculated.

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Average rmse: {:.4f}".format(
                loss, correct, len(test_loader.dataset), percentage_correct, rmse
            )
        )

        if len(active_learning_data.training_dataset) >= max_training_samples:
            break

        N = len(active_learning_data.pool_dataset)
        if acquisition != "random":
            logits_N_K_C = torch.empty(
                (N, num_inference_samples, num_classes),
                dtype=torch.double,
                pin_memory=cuda,
            )

            with torch.no_grad():
                model.eval()

                for i, batch in enumerate(
                    pool_loader_aug
                ):

                    if isinstance(batch, dict):
                        data = batch['image']
                    else:
                        data = batch[0]

                    data = data.to(device=device)

                    data = preprocess_batch(data, vae=vae)

                    lower = i * pool_loader_aug.batch_size
                    upper = min(lower + pool_loader_aug.batch_size, N)
                    logits_N_K_C[lower:upper].copy_(
                        F.log_softmax(model(data, num_inference_samples).double(), -1), non_blocking=True
                    )

            with torch.no_grad():
                candidate_batch = batchbald.get_bald_batch(
                    logits_N_K_C,
                    acquisition_batch_size,
                    # num_samples,#
                    dtype=torch.double,
                    device=device,
                )
                dataset_indices = active_learning_data.get_dataset_indices(
                    candidate_batch.indices
                )
        else:
            # random batch
            indices = torch.randperm(N)[:acquisition_batch_size]
            candidate_batch = namedtuple('Batch','indices')(indices)
            dataset_indices = active_learning_data.get_dataset_indices(
                candidate_batch.indices
            )


        # targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)

        print("Dataset indices: ", dataset_indices)
        if acquisition != "random":
            print("Scores: ", candidate_batch.scores)
        # print("Labels: ", targets[candidate_batch.indices])

        active_learning_data.acquire(candidate_batch.indices)
        active_learning_data_aug.acquire(candidate_batch.indices)
        added_indices.append(dataset_indices)
        # pbar.update(len(dataset_indices))
    # TODO then after that, sanity the pose encoder a bit.
    # TODO add support for preloading instead of training the VAE inline
