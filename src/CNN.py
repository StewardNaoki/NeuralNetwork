import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim

REBUILD_DATA = False # set to true to one once, then back to false unless you want to change something in your training data.
PATH_DATA = "../DATA"
PATH_CAT = "/PetImages/Cat"
PATH_DOG = "/PetImages/Dog"

class DogsVSCats():
    IMG_SIZE = 50
    CATS = PATH_DATA + PATH_CAT
    DOGS = PATH_DATA + PATH_DOG
    TESTING = "../DATA/PetImages/Testing"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []

    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot 
                        #print(np.eye(2)[self.LABELS[label]])

                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1

                    except Exception as e:
                        pass
                        #print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('Cats:',self.catcount)
        print('Dogs:',self.dogcount)




class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

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
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)




def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default = 1,
	            help="number of epoch")
    parser.add_argument("--batch", type=int, default = 100,
	            help="number of batch")
    parser.add_argument("--valpct", type=int, default = 0.1,
	            help="proportion of test data")
    
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
    
    # if(args.test == 1):
    #     testSimpleOLS(args.n, args.sigma)
    # elif(args.test == 2):
    #     evaluateSimpleOLS()
    # elif(args.test == 3):
    #     testSimpleOLSWithAlignedPoints(args.n, args.sigma, args.alignment)
    # elif(args.test == 4):
    #     testRidgeRegression(args.n, args.sigma, args.alignment, getattr(args, 'lambda'))
    # elif(args.test == 5):
    #     testRecursiveLeastSquare(args.n, args.sigma)
    # elif(args.test == 6):
    #     testKalman(args.n, args.sigma, args.drift)
    # plt.show()
    if REBUILD_DATA:
        dogsvcats = DogsVSCats()
        dogsvcats.make_training_data()

    training_data = np.load("training_data.npy", allow_pickle=True)
    print("training data lenght ", len(training_data))


    X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
    X = X/255.0
    y = torch.Tensor([i[1] for i in training_data])

    # plt.imshow(X[0], cmap="gray")
    # plt.show()
    # print(y[0])


    net = Net()
    print("Network architechture:\n",net)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
    X = X/255.0
    y = torch.Tensor([i[1] for i in training_data])

    VAL_PCT = args.valpct  # lets reserve 10% of our data for validation
    val_size = int(len(X)*VAL_PCT)
    print("Lenght of validation data ",val_size)
    train_X = X[:-val_size]
    train_y = y[:-val_size]

    test_X = X[-val_size:]
    test_y = y[-val_size:]
    print(len(train_X), len(test_X))


    BATCH_SIZE = args.batch
    EPOCHS = args.epoch

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            #print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i+BATCH_SIZE]

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()    # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")

    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list, 
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct/total, 3))


if __name__ == "__main__":
    main()