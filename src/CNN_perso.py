import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import time
import sys
# import csv

# set to true to one once, then back to false unless you want to change something in your training data.
CREATE_CSV = True
PATH_DATA = "../DATA"
PATH_CAT = "/PetImages/Cat"
PATH_DOG = "/PetImages/Dog"
MODEL_PATH = "./../log/fc1/best_model.pt"


def make_csv(data_path, list_label_path):
    dict_csv = {'file_name': [],
                'label': []}

    dict_label = {}

    for i in range(len(list_label_path)):
        dict_label[data_path+list_label_path[i]] = i

    print(dict_label)

    for path_label in dict_label:
            print(path_label)
            for f in tqdm(os.listdir(path_label)):
                if "jpg" in f:
                    try:
                        
                        path = os.path.join(path_label, f)
                        X = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if(X is None):
                            print("This image is corrupted: ",path)
                        else:
                            dict_csv['file_name'].append(path)

                            dict_csv['label'].append(np.eye(len(list_label_path))[
                                                    dict_label[path_label]])

                        # dict_csv['label'].append(dict_label[path_label])             
                        #         
                        # print(np.eye(len(list_label_path))[dict_label[path_label]])
                        # print(type(np.eye(len(list_label_path))[dict_label[path_label]]))
                    except Exception as e:
                        pass

    df = pd.DataFrame(dict_csv)
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())
    df.to_csv('catdog.csv', index=False)


class CatDogDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file_name, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file_name)
        self.transform = transform
        self.IMG_SIZE = 28

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # print(self.data_frame.head())

        img_name = os.path.join(self.data_frame.iloc[idx, 0])
        # print("image name: ", img_name)
        X = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        if(X is None):
            print("This image is None: image name: ",img_name)
            assert (not X is None)
        X = cv2.resize(X, (self.IMG_SIZE, self.IMG_SIZE))
        # print(X)

        Y = self.data_frame.iloc[idx, 1]
        Y =Y.replace(" ", ",")
        Y = np.asarray(eval(Y))
        # print(type(Y))
        # print("Y: ",Y)

        # assert (type(Y) == np.int64)

        sample = {'X_image': np.array(X), 'Y': Y}

        if self.transform:
            sample = self.transform(sample)
        
        # return sample
        return (sample['X_image'], sample['Y'])


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, Y = sample['X_image'], sample['Y']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print(type(image.shape))
        # print(len(image.shape))
        # print(Y)

        
        if len(image.shape) == 3:
            print(image.shape)
            image = image.transpose((2, 0, 1))
        elif len(image.shape) == 2:
            image = np.array([image])

        # landmarks = landmarks.transpose((2, 0, 1))
        # return {'X_image': torch.from_numpy(image),
        #         'Y': torch.from_numpy(Y).long()}

        return {'X_image': torch.from_numpy(image),
                'Y': torch.from_numpy(Y).float()}

        # print("Y torch ", Y)
        # print(torch.LongTensor(np.array([Y])))

        # return {'X_image': torch.from_numpy(image),
        #         'Y': torch.LongTensor(np.array([Y])).float()}
    
class Normalize(object):

    def __call__(self, sample):
        image = sample['X_image']

        # landmarks = landmarks.transpose((2, 0, 1))
        return {'X_image': (image/255) -0.5,
                'Y': sample['Y']}

class CrossEntropyOneHot(object):


    def __call__(self, sample):
        _, labels = sample['Y'].max(dim=0)
        # landmarks = landmarks.transpose((2, 0, 1))
        return {'X_image': sample['X_image'],
                'Y': labels}



def compute_mean_std(loader):
    # Compute the mean over minibatches
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    # Compute the std over minibatches
    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img)**2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)

    # Set the variance of pixels with no variance to 1
    # Because there is no variance
    # these pixels will anyway have no impact on the final decision
    std_img[std_img == 0] = 1

    return mean_img, std_img

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=0)
    return nn.CrossEntropyLoss()(input, labels)


class Net(nn.Module):
    def __init__(self):
        super().__init__()  # just run the init of parent class (nn.Module)
        # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv1 = nn.Conv2d(1, 32, 5)
        # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        # 512 in, 2 out bc we're doing 2 classes (dog vs cat).
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        # .view is reshape ... this flattens X before
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.l2_reg = 0.001

        self.conv1 = nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            )

        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)

        self.layer1 = nn.Sequential(         # input shape (1, 28, 28)
            self.conv1,                     # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.layer2 = nn.Sequential(         # input shape (16, 14, 14)
            self.conv2,                     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 2)   # fully connected layer, output 10 classes

    def penalty(self):
        return self.l2_reg * (self.conv1.weight.norm(2) + self.conv2.weight.norm(2))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        # return output, x    # return x for visualization
        return F.softmax(output, dim = 1)
    
def train(model, loader, f_loss, optimizer, device):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()
    
    N = 0
    tot_loss, correct = 0.0, 0.0
    
    for i, (inputs, targets) in tqdm(enumerate(loader)):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)

        # print("target ",targets)
        # print("output ",outputs)

        # _, target_entropy = targets.max(dim = 0)
        # outputs = outputs.float()
        targets = targets.float()
        # _, output_max = outputs.max(dim=0)
        _, target_max = targets.max(dim=1)

        # print("target ",target_max)
        # print("output ",outputs)

        loss = f_loss(outputs, target_max)
        N += inputs.shape[0]
        tot_loss += inputs.shape[0] * f_loss(outputs, target_max).item()
        # tot_loss += inputs.shape[0] * f_loss(outputs, ).item()
        # print("Output: ", outputs)
        predicted_targets = outputs.argmax(dim=1)
        # targets = targets.argmax(dim=1)
        # print("Predicted target ",predicted_targets)
        # print("target ",targets)
        correct += (predicted_targets == target_max).sum().item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        model.penalty().backward()
        optimizer.step()
        
    return tot_loss/N, correct/N
    

def test(model, loader, f_loss, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation 

    Returns :

        A tuple with the mean loss and mean accuracy

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        for i, (inputs, targets) in tqdm(enumerate(loader)):

            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            _, target_max = targets.max(dim=1)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            tot_loss += inputs.shape[0] * f_loss(outputs, target_max).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs.argmax(dim=1)
            targets = targets.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()
        return tot_loss/N, correct/N

class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model to ", self.filepath)
            torch.save(self.model.state_dict(), self.filepath)
            #torch.save(self.model, self.filepath)
            self.min_loss = loss

def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

def progress(loss, acc):
    print(' Training   : Loss : {:2.4f}, Acc : {:2.4f}\r'.format(loss, acc))
    sys.stdout.flush()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1,
                        help="number of epoch (default: 1)")
    parser.add_argument("--batch", type=int, default=100,
                        help="number of batch (default: 100)")
    parser.add_argument("--valpct", type=float, default=0.2,
                        help="proportion of test data (default: 0.2)")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="number of thread used (default: 1)")
    parser.add_argument("--create_csv", type=bool, default=False,
                        help="create or not csv file (default: False)")

    # parser.add_argument("--n", type=int, default = 100,
#             help="number of generated samples")
    # parser.add_argument("--sigma", type=float, default = 1.,
#             help="standard deviation of error")
    # parser.add_argument("--alignment", type=float, default = 0.1,
#             help="alignment factor")
    # parser.add_argument("--lambda", type=float, default = 1.,
#             help="lambda factor")
    # parser.add_argument("--drift", type=float, default = 1.,
#             help="model coefficient drift (standard deviation)")
    args = parser.parse_args()

    if args.create_csv:
        make_csv(PATH_DATA, [PATH_CAT, PATH_DOG])

    valid_ratio = args.valpct  # Going to use 80%/20% split for train/valid

    data_transforms = transforms.Compose([
        ToTensor(), Normalize()
        ])

    full_dataset = CatDogDataset(
        csv_file_name='catdog.csv', transform=data_transforms)

    nb_train = int((1.0 - valid_ratio) * len(full_dataset))
    # nb_test = int(valid_ratio * len(full_dataset))
    nb_test = len(full_dataset) - nb_train
    print("Size of full data set: ",len(full_dataset))
    print("Size of training data: ", nb_train)
    print("Size of testing data: ", nb_test)
    train_dataset, test_dataset = torch.utils.data.dataset.random_split(
        full_dataset, [nb_train, nb_test])

    # print("Test lenght: ", len(train_dataset))
    # print("Test lenght: ", len(test_dataset))

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=args.batch,
    #                                            shuffle=True,
    #                                            num_workers=args.num_threads)

    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=args.batch,
    #                                            shuffle=True,
    #                                            num_workers=args.num_threads)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            num_workers=args.num_threads)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            num_workers=args.num_threads)

    # for (inputs, targets) in train_loader:
    #     # print("input:\n",input)
    #     print("target\n", targets)

    #     break

    # model = Net()
    model  = CNN()
    print("Network architechture:\n",model)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    f_loss = torch.nn.CrossEntropyLoss()
    # f_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())


    top_logdir = './../log/fc1'
    if not os.path.exists(top_logdir):
        os.mkdir(top_logdir)
    model_checkpoint = ModelCheckpoint(top_logdir + "/best_model.pt", model)

    for t in tqdm(range(args.epoch)):
        print("Epoch {}".format(t))
        train_loss, train_acc = train(model, train_loader, f_loss, optimizer, device)
        
        progress(train_loss, train_acc)
        time.sleep(0.5)
        
        val_loss, val_acc = test(model, test_loader, f_loss, device)
        print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))
        
    #    logdir = generate_unique_logpath(top_logdir, "linear")
    #    print("Logging to {}".format(logdir))
    #    # -> Prints out     Logging to   ./logs/linear_0
    #    model_checkpoint.filepath = logdir
    #    if not os.path.exists(logdir):
    #        os.mkdir(logdir)
        model_checkpoint.update(val_loss)
        
        # tensorboard_writer.add_scalar('metrics/train_loss', train_loss, t)
        # tensorboard_writer.add_scalar('metrics/train_acc',  train_acc, t)
        # tensorboard_writer.add_scalar('metrics/val_loss', val_loss, t)
        # tensorboard_writer.add_scalar('metrics/val_acc',  val_acc, t)

    model.load_state_dict(torch.load(MODEL_PATH))
    test_loss, test_acc = test(model, test_loader, f_loss, device)
    print(" Test       : Loss : {:.4f}, Acc : {:.4f}".format(test_loss, test_acc))


if __name__ == "__main__":
    main()
