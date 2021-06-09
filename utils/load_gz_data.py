import os

import pandas as pd
from PIL import Image
import torch
import torchvision as tv

class Gz2_data(torch.utils.data.Dataset):
    def __init__(self, csv_dir, image_dir, list_of_interest, faulty_data_set=False, crop=180,
                 resize=128, transforms=None, one_hot_categorical=False, data_aug=False):
        self.csv_dir = csv_dir
        self.faulty_data_set = faulty_data_set
        self.image_dir = image_dir
        self.file = pd.read_csv(self.csv_dir)
        self.list_of_interest = list_of_interest
        self.resize = resize
        self.crop = crop
        self.one_hot_categorical = one_hot_categorical
        self.data_aug = data_aug
    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.faulty_data_set == True:
            image = None
            while image is None:
                try:
                    img_name = os.path.join(self.image_dir, self.file['png_loc'][idx])
                    img_name = img_name.replace(".png", ".jpg")
                    image = Image.open(img_name)
                except FileNotFoundError:
                    idx += 1
        else:
            img_name = os.path.join(self.image_dir, self.file['png_loc'][idx])
            img_name = img_name.replace(".png", ".jpg")
            image = Image.open(img_name).convert('L')

        data = self.file.iloc[idx][self.list_of_interest]
        data = torch.tensor(data.values.astype('float'))
        if self.data_aug:
            transforms = tv.transforms.Compose(
            [tv.transforms.CenterCrop(self.crop),
             tv.transforms.Resize(self.resize), tv.transforms.Grayscale(),
             tv.transforms.RandomRotation(180),
             tv.transforms.ToTensor()])
        else:
            transforms = tv.transforms.Compose(
                [tv.transforms.CenterCrop(self.crop),
                 tv.transforms.Resize(self.resize), tv.transforms.Grayscale(),
                 #             tv.transforms.RandomRotation(180), tv.transforms.RandomAffine(180),
                 tv.transforms.ToTensor()])
        if self.one_hot_categorical == True:
            _, indice = data.max(0)
            out = torch.zeros(len(data))
            out[indice] = 1.0
            data = out
        image = transforms(image)
        
        sample = {'image': image, 'data': data.float()}
        return sample

def return_data_loader(data, test_proportion, batch_size, shuffle=True):
    len_data = len(data)
    num_tests = int(len_data * test_proportion)
    test_indices = list(i for i in range(0,num_tests))
    train_indices = list(i for i in range(num_tests, len_data))
    test_set = torch.utils.data.Subset(data, test_indices)
    train_set = torch.utils.data.Subset(data, train_indices)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=shuffle)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader
    
def return_ss_loader(data, test_proportion, supervised_portion,
                     batch_size, shuffle=True, subset_unsupervised_proportion=None):
    if subset_unsupervised_proportion != None:
        if subset_unsupervised_proportion < 1.0:
            len_data = len(data)*subset_unsupervised_proportion
        else:
            len_data = int(subset_unsupervised_proportion)
    if supervised_portion < 1.0:
        us_portion = 1.0 - supervised_portion
        num_tests = round(len_data * test_proportion)
        num_train = len_data - num_tests
        us_tests = round(num_tests * us_portion)
        s_tests = num_tests - us_tests
        us_train = round(num_train * us_portion)
        s_train = num_train - us_train
        us_train = num_train - us_tests

    else:
        len_data = len(data)
        s_train = int(supervised_portion)
        remain = len_data - supervised_portion
        num_tests = round(remain * test_proportion)
        num_train = len_data - num_tests

        s_tests = round(supervised_portion * test_proportion)
        us_tests = num_tests - s_tests
        us_train = round(remain - s_tests - us_tests)

    test_supervised = list(i for i in range(0,s_tests))
    test_unsup = list(i for i in range(s_tests, num_tests))
    
    train_supervised = list(i for i in range(num_tests, num_tests + s_train))
    train_unsup = list(i for i in range(num_tests + s_train , len_data))

    test_s_set = torch.utils.data.Subset(data, test_supervised)
    test_us_set = torch.utils.data.Subset(data, test_unsup)
    train_s_set = torch.utils.data.Subset(data, train_supervised)
    train_us_set = torch.utils.data.Subset(data, train_unsup)
    test_s_loader = torch.utils.data.DataLoader(dataset=test_s_set, batch_size=batch_size, shuffle=shuffle)
    test_us_loader = torch.utils.data.DataLoader(dataset=test_us_set, batch_size=batch_size, shuffle=shuffle)
    train_s_loader = torch.utils.data.DataLoader(dataset=train_s_set, batch_size=batch_size, shuffle=shuffle)
    train_us_loader = torch.utils.data.DataLoader(dataset=train_us_set, batch_size=batch_size, shuffle=shuffle)

    return test_s_loader, test_us_loader, train_s_loader, train_us_loader


def return_subset(data, test_proportion, subset_portion, batch_size, shuffle=False):

    if subset_portion > 1.0:
        num_data = int(subset_portion)
    else:
        num_data = int(len(data)*subset_portion)
    
    num_tests = int(num_data * test_proportion)
    test_indices = list(i for i in range(0, num_tests))
    train_indices = list(i for i in range(num_tests, num_data))
    test_set = torch.utils.data.Subset(data, test_indices)
    train_set = torch.utils.data.Subset(data, train_indices)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader

if __name__ == "__main__":
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
    # not sure why PIL doesnt work with ~
    data = Gz2_data(csv_dir="~/diss/gz2_data/gz_amended.csv",
                    image_dir="/Users/Mizunt/diss/gz2_data",
                    list_of_interest=[a01,
                                      a02,
                                      a03], faulty_data_set=False)
    sample_of_data = data[1]
    print(type(sample_of_data['data']))
    print(sample_of_data['image'].shape)
    print(type(sample_of_data['image']))
    train_loader, test_loader = return_data_loader(data, 0.2, 20)
