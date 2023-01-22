import os
from notebook_utils import *
import random
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim


CLASSES = ['rock', 'paper', 'scissors']
model_input_size = (64, 64)


file_pattern_training_data = '/Users/amling/uni/shifumi/DataEng/datasets/combined/combined/*/*.png'
file_pattern_validation_data = '/Users/amling/uni/shifumi/DataEng/datasets/xAI-Proj-M-validation_set_pp_01_grey/*/*.png'
file_pattern_test_data = '/Users/amling/uni/shifumi/DataEng/datasets/xAI-Proj-M-validation_set_pp_01_grey/*/*.png'

class RPS_CNN(nn.Module):
    def __init__(self, activate_dropout: bool = True, dropout_probability: float = 0.5,
                 batch_normalization: bool = False, convolution_kernel_size: int = 3):
        super(RPS_CNN, self).__init__()
        self.activate_dropout = activate_dropout
        self.dropout_probability = dropout_probability
        self.batch_normalization = batch_normalization

        # first block of convolutional layers
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=convolution_kernel_size, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=convolution_kernel_size, padding=1)
        # add a convolutional layer 32x32
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=convolution_kernel_size, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=convolution_kernel_size, padding=1)
        # add Convolution Layer 64x64
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=convolution_kernel_size, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=convolution_kernel_size, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=convolution_kernel_size, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=convolution_kernel_size, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=convolution_kernel_size, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=convolution_kernel_size, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.adaptive_maxpool = nn.AdaptiveMaxPool2d(2)

        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 3)

    def forward(self, x):
        # feature extraction part
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        if self.batch_normalization:
            x = self.bn2(x)
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        if self.batch_normalization:
            x = self.bn3(x)
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        if self.batch_normalization:
            x = self.bn4(x)
        x = self.adaptive_maxpool(x)
        # actual classification part
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.activate_dropout:
            x = F.dropout(x, self.dropout_probability)
        x = F.relu(self.fc2(x))
        if self.activate_dropout:
            x = F.dropout(x, self.dropout_probability)
        x = self.fc3(x)
        return x


def test_accuracy(model, testloader):
    # Test the model on the test dataset
    print("test accuracy called")
    outcome = {}
    with torch.no_grad():
        correct = 0
        total = 0
        class_correct = [0. for i in range(len(CLASSES))]
        class_total = [0. for i in range(len(CLASSES))]
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        print(f'Accuracy of the network on the test set: {100 * correct / total}%')
        outcome['total'] = 100 * correct / total
        for i in range(len(CLASSES)):
            if class_total[1] > 0:
                print(f'Accuracy of {CLASSES[i]} : {100 * class_correct[i] / class_total[i]}%')
                outcome[CLASSES[i]] = 100 * class_correct[i] / class_total[i]
            else:
                print(f'Accuracy of {CLASSES[i]} : could not be measured, no test images%')
    return outcome


def train(model, epoches, training_data, val_data, save_model_each_x_epoches=10):
    print("training starts")
    hist = []
    for epoch in range(epoches):
        print(f"epoch {epoch + 1} started")
        running_loss = 0.0
        for i, data in enumerate(training_data, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch + 1} loss: {epoch_loss}')
        hist.append({'loss': epoch_loss, 'accuracy_measures': test_accuracy(val_data)})
        if save_model_each_x_epoches > 0:
            if (epoch + 1) % save_model_each_x_epoches == 0:
                name = f'pytk_rock_paper_scissors_{epoch + 1}epoches__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}'
                model_save_path = f'./model_states/{name}.pt'
                torch.save(model.state_dict(), model_save_path)
                hist_save_path = f'./model_states/history__{name}'
                save_history(hist_save_path, hist)
    return hist


class RPSDataset(Dataset):
    """
    our custom Rock Paper Scissors Dataset
    """

    def __init__(self, x, y, transforms=None):
        self.x = x
        self.y = torch.LongTensor(y)
        self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        if self.transforms is not None:
            x = self.transforms(x)
        return x, y


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(0, shear=0.2),  # random shear 0.2
    transforms.RandomAffine(0, scale=(0.8, 1.2)),  # random zoom 0.2
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

image_transforms = transforms.Compose([
    transforms.ToTensor()
])

if __name__ == "__main__":

    MODEL_SAVE_DIR = os.path.join('.', 'model_states')

    if not os.path.exists(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    assert os.path.exists(MODEL_SAVE_DIR)
    print('MODEL_SAVE_DIR = %s' % MODEL_SAVE_DIR)

    import warnings

    # warnings.filterwarnings('ignore')

    # tweaks for libraries
    np.set_printoptions(precision=6, linewidth=1024, suppress=True)

    # Pytorch imports
    gpu_available = torch.cuda.is_available()
    print('Using Pytorch version: %s. GPU %s available' % (torch.__version__, "IS" if gpu_available else "is NOT"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED);

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False





    train_images, train_labels, train_data_paths = get_images_and_labels_from_filepattern(file_pattern_training_data,
                                                                                          model_input_size)
    val_images, val_labels, val_data_paths = get_images_and_labels_from_filepattern(file_pattern_validation_data,
                                                                                    model_input_size)
    test_images, test_labels, test_data_paths = get_images_and_labels_from_filepattern(file_pattern_test_data,
                                                                                       model_input_size)

    print(f" training images.shape: {train_images.shape} - labels.shape: {train_labels.shape}")
    print(f"training data from {train_data_paths[:2]}\n")
    print(f" val images.shape: {val_images.shape} - labels.shape: {val_labels.shape}")
    print(f"val data from {val_data_paths[:2]}\n")
    print(f" test images.shape: {test_images.shape} - labels.shape: {test_labels.shape}")
    print(f"test data from {test_data_paths[:2]}\n")

    # definition of parameters
    batch_size = 64
    training_epoches = 100
    use_batch_normalization = False
    use_dropouts = True
    dropout_probability = 0.5
    learning_rate = 1e-4

    print("parameters set")
    # %%
    train_dataset = RPSDataset(train_images, train_labels, train_transforms)
    val_dataset = RPSDataset(val_images, val_labels, image_transforms)
    test_dataset = RPSDataset(test_images, test_labels, image_transforms)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # %%
    # look a train dataset
    show_images_from_dataloader(trainloader)

    # look a validation dataset
    show_images_from_dataloader(validationloader)

    # Define the model, loss function, and optimizer
    model = RPS_CNN(activate_dropout=use_dropouts, dropout_probability=dropout_probability,
                    batch_normalization=use_batch_normalization)
    model = model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("model crteated")

    import time

    start_time = time.time()
    hist = train(model, epoches=training_epoches, training_data=trainloader, val_data=validationloader)
    training_time_s = time.time() - start_time

    print('testing against the validation dataset')
    test_accuracy(model,validationloader)

    print('testing against the testing dataset')
    test_accuracy(model, testloader)

    print(f'training took {convert_seconds(training_time_s)}')
    print(model)

    # plot the validation loss during training
    show_validation_loss(hist,
                         save_path=f'./model_states/{training_epoches}epoches_val_loss__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}.png')
    # plot the validation loss
    show_training_accuracy(hist,
                           save_path=f'./model_states/{training_epoches}epoches_accuracy__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}.png')
