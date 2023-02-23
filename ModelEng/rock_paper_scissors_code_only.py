import os

# set file path as working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

file_pattern_training_data = 'C:/Users/bapti/Documents/Data/combined_grey/combined/*/*.png'
file_pattern_validation_data = 'C:/Users/bapti/Documents/Data/xAI-Proj-M-validation_set_grey/*/*.png'
file_pattern_test_data = 'C:/Users/bapti/Documents/Data/xAI-Proj-M-testing_set_grey/*/*.png'


class RPS_CNN(nn.Module):
    def __init__(self, name, activate_dropout: bool = True, dropout_probability: float = 0.5,
                 batch_normalization: bool = False, convolution_kernel_size: int = 3):
        super(RPS_CNN, self).__init__()
        self.name = name
        self.activate_dropout = activate_dropout
        self.dropout_probability = dropout_probability
        self.batch_normalization = batch_normalization

        # Define the first block of convolutional layers with specified kernel size and padding
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=convolution_kernel_size, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=convolution_kernel_size, padding=1)
        # Add batch normalization layer after the first block of convolutional layers
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=convolution_kernel_size, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=convolution_kernel_size, padding=1)
        # Add batch normalization layer after the second block of convolutional layers
        self.bn2 = nn.BatchNorm2d(64)

        # Define the second block of convolutional layers
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=convolution_kernel_size, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=convolution_kernel_size, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=convolution_kernel_size, padding=1)
        # Add batch normalization layer after the second block of convolutional layers
        self.bn3 = nn.BatchNorm2d(128)

        # Define the third block of convolutional layers
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=convolution_kernel_size, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=convolution_kernel_size, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=convolution_kernel_size, padding=1)
        # Add batch normalization layer after the third block of convolutional layers
        self.bn4 = nn.BatchNorm2d(256)

        # Define maxpooling and adaptive maxpooling layers
        self.maxpool = nn.MaxPool2d(2, 2)
        self.adaptive_maxpool = nn.AdaptiveMaxPool2d(2)

        # Define fully connected layers for classification
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 3)

    def forward(self, x):
        # Feature extraction part
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
        # Actual classification part
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.activate_dropout:
            x = F.dropout(x, self.dropout_probability)
        x = F.relu(self.fc2(x))
        if self.activate_dropout:
            x = F.dropout(x, self.dropout_probability)
        x = self.fc3(x)
        return x


def test_accuracy(model, testloader, criterion=None):
    """
    Evaluate the accuracy of a model on a given test dataset.

    Args:
        model (torch.nn.Module): the model to evaluate.
        testloader (torch.utils.data.DataLoader): the data loader for the test dataset.
        criterion (callable, optional): a criterion to calculate the loss. Default: None.
        classes (list of str, optional): the names of the classes. Default: None.

    Returns:
        A tuple (outcome, avg_loss), where outcome is a dictionary with the accuracy for each class
        and the total accuracy, and avg_loss is the average loss over the test dataset.
    """
    print("Test accuracy called")
    outcome = {}
    running_loss = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        class_correct = [0. for i in range(len(CLASSES))]
        class_total = [0. for i in range(len(CLASSES))]
        for inputs, labels in testloader:
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        print(f'Accuracy of the network on the test set: {100 * correct / total}%')
        avg_loss = running_loss/len(testloader)
        print(f'Loss of the network on the test set: {avg_loss}')
        outcome['total'] = 100 * correct / total
        for i in range(len(CLASSES)):
            if class_total[1] > 0:
                print(f'Accuracy of {CLASSES[i]} : {100 * class_correct[i] / class_total[i]}%')
                outcome[CLASSES[i]] = 100 * class_correct[i] / class_total[i]
            else:
                print(f'Accuracy of {CLASSES[i]} : could not be measured, no test images%')
    return outcome, avg_loss


def train(model, epoches, optimizer, criterion, training_data, val_data, save_model_each_x_epoches=10):
    """
    Train a PyTorch model on a training dataset, and evaluate it on a validation dataset.
    Saves the model state and training history periodically.
    Returns a list of dictionaries containing information about each epoch.
    """
    print("Training starts")
    hist = []
    for epoch in range(epoches):
        print(f"Epoch {epoch + 1} started")
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0. for i in range(len(CLASSES))]
        class_total = [0. for i in range(len(CLASSES))]

        # Iterate over the training data
        for i, data in enumerate(training_data, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Compute training accuracy and epoch loss
        training_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(training_data)
        print(f'Epoch {epoch + 1} loss: {epoch_loss}')

        # Evaluate on the validation data
        val_outcome, val_loss = test_accuracy(model, val_data, criterion)

        # Record epoch history
        hist.append({'loss': epoch_loss, 'val_loss': val_loss, 'accuracy_measures': val_outcome, 'training_accuracy' : training_accuracy})
        
        # Save model and history periodically
        if save_model_each_x_epoches > 0:
            if (epoch + 1) % save_model_each_x_epoches == 0:
                name = f'{model.name}_{epoch + 1}epoches__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}'
                model_save_path = f'model_states/model_state/{name}.pt'
                torch.save(model.state_dict(), model_save_path)
                hist_save_path = f'{os.getcwd()}/model_states/history__{name}'
                save_history(hist_save_path, hist)
    return hist


class RPSDataset(Dataset):
    """
    Custom Rock Paper Scissors dataset
    """

    def __init__(self, x, y, transforms=None):
        """
        Constructor for RPSDataset class.

        Args:
        - x (list): List of images (numpy arrays) 
        - y (list): List of labels (integers)
        - transforms (callable): Optional transforms to be applied to the images
        """
        self.x = x
        self.y = torch.LongTensor(y)
        self.transforms = transforms

    def __len__(self):
         """
        Returns the length of the dataset.
        """
        return len(self.x)

    def __getitem__(self, ix):
         """
        Returns the item at the given index.

        Args:
        - ix (int): Index of the item to retrieve.

        Returns:
        - tuple: A tuple containing the transformed image (tensor) and its corresponding label (int).
        """
        x, y = self.x[ix], self.y[ix]
        if self.transforms is not None:
            x = self.transforms(x)
        return x, y


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(0, shear=0.2),
    transforms.RandomAffine(0, scale=(0.8, 1.2)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

image_transforms = transforms.Compose([
    transforms.ToTensor()
])




def experiment():
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
    model_name = 'baptiste'
    batch_size = 64
    training_epoches = 100
    dropout_probability = 0.5
    learning_rate = 1e-4
    print("parameters set")
    
    train_dataset = RPSDataset(train_images, train_labels, train_transforms)
    val_dataset = RPSDataset(val_images, val_labels, image_transforms)
    test_dataset = RPSDataset(test_images, test_labels, image_transforms)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # look a train dataset
    show_images_from_dataloader(trainloader)

    # look a validation dataset
    show_images_from_dataloader(validationloader)
    
    import time
    
    for use_batch_normalization in [True, False]:
        for use_dropouts in [True, False]:
            # Define the model, loss function, and optimizer
            model = RPS_CNN(name=model_name, activate_dropout=use_dropouts, dropout_probability=dropout_probability,
                            batch_normalization=use_batch_normalization)
            model = model.to(device=device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            print("model created")
            
            
            start_time = time.time()
            hist = train(model, optimizer = optimizer, criterion = criterion, epoches=training_epoches, training_data=trainloader, val_data=validationloader)
            training_time_s = time.time() - start_time

            print('testing against the validation dataset')
            test_accuracy(model, validationloader)

            print('testing against the testing dataset')
            test_accuracy(model, testloader)

            print(f'training took {convert_seconds(training_time_s)}')
            print(model)

            # plot the validation loss during training
            show_validation_loss(hist,
                                 save_path=f'./model_states/{model_name}_{training_epoches}epoches_val_loss__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}.png')
            # plot the validation loss
            show_training_accuracy(hist,
                                   save_path=f'./model_states/{model_name}_{training_epoches}epoches_accuracy__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}.png')
            

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
    model_name = 'abc'
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
    model = RPS_CNN(name=model_name, activate_dropout=use_dropouts, dropout_probability=dropout_probability,
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
    test_accuracy(model, validationloader)
    print('testing against the testing dataset')
    test_accuracy(model, testloader)
    print(f'training took {convert_seconds(training_time_s)}')
    print(model)
    # plot the validation loss during training
    show_validation_loss(hist,
                         save_path=f'./model_states/{model_name}_{training_epoches}epoches_val_loss__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}.png')
    # plot the validation loss
    show_training_accuracy(hist,
                           save_path=f'./model_states/{model_name}_{training_epoches}epoches_accuracy__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}.png')