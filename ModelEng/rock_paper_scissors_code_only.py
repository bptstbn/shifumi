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

file_pattern_training_data = '/home/divingsoup/Documents/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/combined/*/*.png'
file_pattern_validation_data = '/home/divingsoup/Documents/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/validation_grey_pp01/*/*.png'
file_pattern_test_data = '/home/divingsoup/Documents/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/testing_grey_pp01/*/*.png'

file_pattern_validation_data_rmNoise = '/home/divingsoup/Documents/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/validation_grey_pp01_randomNoise/*/*.png'
file_pattern_validation_data_gaussNoise = '/home/divingsoup/Documents/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/validation_grey_pp01_gaussianNoise/*/*.png'

file_pattern_test_data_rmNoise = '/home/divingsoup/Documents/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/testing_grey_pp01_rm_noise/*/*.png'
file_pattern_test_data_gaussNoise = '/home/divingsoup/Documents/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/testing_grey_pp01_gaussian_noise/*/*.png'


class RPS_CNN(nn.Module):
    def __init__(self, name, activate_dropout: bool = True, dropout_probability: float = 0.5,
                 batch_normalization: bool = False, convolution_kernel_size: int = 3):
        super(RPS_CNN, self).__init__()
        self.name = name
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


def test_accuracy(model, testloader, criterion=None):
    # Test the model on the test dataset
    #print("test accuracy called")
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
        #print(f'Loss of the network on the test set: {running_loss/len(testloader)}')
        outcome['total'] = 100 * correct / total
        for i in range(len(CLASSES)):
            if class_total[1] > 0:
                print(f'Accuracy of {CLASSES[i]} : {100 * class_correct[i] / class_total[i]}%')
                outcome[CLASSES[i]] = 100 * class_correct[i] / class_total[i]
            else:
                print(f'Accuracy of {CLASSES[i]} : could not be measured, no test images%')
    return outcome, running_loss/len(testloader)


def train(model, epoches, optimizer, criterion, training_data, val_data, save_model_each_x_epoches=10):
    print("training starts")
    hist = []
    for epoch in range(epoches):
        print(f"epoch {epoch + 1} started")
        running_loss = 0.0
        
        ###
        correct = 0
        total = 0
        class_correct = [0. for i in range(len(CLASSES))]
        class_total = [0. for i in range(len(CLASSES))]
        ####
        
        for i, data in enumerate(training_data, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            
            ###
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            ###
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        ###
        training_accuracy = 100 * correct / total
        ###
        epoch_loss = running_loss / len(training_data)
        print(f'Epoch {epoch + 1} loss: {epoch_loss}')
        val_outcome, val_loss = test_accuracy(model, val_data, criterion)
        hist.append({'loss': epoch_loss, 'val_loss': val_loss, 'accuracy_measures': val_outcome, 'training_accuracy' : training_accuracy})
        if save_model_each_x_epoches > 0:
            if (epoch + 1) % save_model_each_x_epoches == 0:
                """
                name = f'./model_state/{model.name}_{epoch + 1}epoches__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}'
                model_save_path = f'./model_states/{name}.pt'
                torch.save(model.state_dict(), model_save_path)
                hist_save_path = f'./model_states/history__{name}'
                save_history(hist_save_path, hist)"
                """
                name = f'{model.name}_{epoch + 1}epoches__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}'
                model_save_path = f'model_states/model_state/{name}.pt'
                torch.save(model.state_dict(), model_save_path)
                hist_save_path = f'{os.getcwd()}/model_states/history__{name}'
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
    transforms.RandomAffine(0, shear=0.2),
    transforms.RandomAffine(0, scale=(0.8, 1.2)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

image_transforms = transforms.Compose([
    transforms.ToTensor()
])


class RandomPixelRemoval(object):
    def __init__(self, p=0.3, remove_pct=0.25):
        self.p = p
        self.remove_pct = remove_pct

    def __call__(self, image):
        if random.uniform(0, 1) > self.p:
            return image

        h, w, _ = image.shape
        num_pixels_to_remove = int(h * w * self.remove_pct)

        # Remove random pixels
        for i in range(num_pixels_to_remove):
            row = random.randint(0, h-1)
            col = random.randint(0, w-1)
            image[row, col] = 0

        return image


def remove_parts_transformation(image, prob=0.25):
    # Generate a random mask
    mask = np.random.rand(image.shape[0], image.shape[1]) < prob
    mask = torch.from_numpy(mask)
    image[mask] = torch.tensor([0, 0, 0], dtype=torch.float32)
    return image



class WithNoiseTransform(object):
    def __init__(self, normal_prob=0.7):
        self.normal_prob = normal_prob

    def __call__(self, image):
        if random.random() > self.normal_prob:
            return remove_parts_transformation(image)
        else:
            return image


train_withNoise_transforms = transforms.Compose([
    RandomPixelRemoval(),
    transforms.ToTensor(),
    transforms.RandomAffine(0, shear=0.2),
    transforms.RandomAffine(0, scale=(0.8, 1.2)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

train_withNoise_transforms2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=1, sigma=25),
    transforms.RandomAffine(0, shear=0.2),
    transforms.RandomAffine(0, scale=(0.8, 1.2)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
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

            print('testing against the validation dataset:')
            test_accuracy(model, validationloader)

            print('testing against the testing dataset:')
            test_accuracy(model, testloader)

            print(f'training took {convert_seconds(training_time_s)}')
            print(model)

            # plot the validation loss during training
            show_validation_loss(hist,
                                 save_path=f'./model_states/{model_name}_{training_epoches}epoches_val_loss__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}.png')
            # plot the validation loss
            show_training_accuracy(hist,
                                   save_path=f'./model_states/{model_name}_{training_epoches}epoches_accuracy__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}.png')

"""     
# experiment()
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
    model_name = 'noisy_train'
    batch_size = 64
    training_epoches = 100
    use_batch_normalization = False
    use_dropouts = True
    dropout_probability = 0.5
    learning_rate = 1e-4

    print("parameters set")
    # %%
    train_dataset = RPSDataset(train_images, train_labels, train_withNoise_transforms)
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
    hist = train(model, epoches=training_epoches, optimizer=optimizer, criterion=criterion, training_data=trainloader, val_data=validationloader)
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
    #show_training_accuracy(hist,
     #                      save_path=f'./model_states/{model_name}_{training_epoches}epoches_accuracy__Dropouts_{str(model.activate_dropout)}__BatchNorm_{str(model.batch_normalization)}.png')


"""
def get_all_predictions(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            , dim=0
        )
    return all_preds


from sklearn.metrics import confusion_matrix


def print_confusion_matrices(dataset):
    with torch.no_grad():
        prediction_loader = DataLoader(dataset, batch_size=10000)
        train_preds = get_all_predictions(model, prediction_loader)

    truthLoader = DataLoader(dataset, batch_size=10000, shuffle=False)

    test_targets_list = []
    for inputs, targets in truthLoader:
        test_targets_list.append(targets)

    test_targets = test_targets_list[0]
    print(len(test_targets))
    print(len(train_preds.argmax(dim=1)))

    cm = confusion_matrix(test_targets, train_preds.argmax(dim=1))

    # Print the confusion matrix using Matplotlib
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.0f')

    ax.set_title('RPS Classification\n\n');
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Rock', 'Paper', 'Scissors'])
    ax.yaxis.set_ticklabels(['Rock', 'Paper', 'Scissors'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()

    # Display matrix in percent
    ax = sns.heatmap(cm / np.sum(cm), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_title('RPS Classification\n\n');
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Rock', 'Paper', 'Scissors'])
    ax.yaxis.set_ticklabels(['Rock', 'Paper', 'Scissors'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


def get_all_predictions_new(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            , dim=0
        )
    return all_preds


def print_confusion_matrices_new(model, dataset, batch_size):
    with torch.no_grad():
        total_samples = len(dataset)
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        for i in range(0, total_samples, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, total_samples)
            batch_dataset = torch.utils.data.Subset(dataset, range(batch_start, batch_end))
            prediction_loader = DataLoader(batch_dataset, batch_size=batch_size)
            train_preds = get_all_predictions_new(model, prediction_loader)

            truth_loader = DataLoader(batch_dataset, batch_size=batch_size, shuffle=False)

            test_targets_list = []
            for inputs, targets in truth_loader:
                test_targets_list.append(targets)

            test_targets = test_targets_list[0]

            all_preds = torch.cat((all_preds, train_preds.argmax(dim=1)))
            all_labels = torch.cat((all_labels, test_targets))

    cm = confusion_matrix(all_labels, all_preds)

    # Print the confusion matrix using Matplotlib
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.0f')

    ax.set_title('RPS Classification\n\n');
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Rock', 'Paper', 'Scissors'])
    ax.yaxis.set_ticklabels(['Rock', 'Paper', 'Scissors'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


    # Display matrix in percent
    ax = sns.heatmap(cm / np.sum(cm), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_title('RPS Classification\n\n');
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Rock', 'Paper', 'Scissors'])
    ax.yaxis.set_ticklabels(['Rock', 'Paper', 'Scissors'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


if __name__ == '__main__':

    train_images, train_labels, train_data_paths = get_images_and_labels_from_filepattern(file_pattern_training_data,
                                                                                          model_input_size)
    val_images, val_labels, val_data_paths = get_images_and_labels_from_filepattern(file_pattern_validation_data,
                                                                                    model_input_size)
    test_images, test_labels, test_data_paths = get_images_and_labels_from_filepattern(file_pattern_test_data,
                                                                                       model_input_size)
    val_rm_images, val_rm_labels, val_rm_data_paths = get_images_and_labels_from_filepattern(file_pattern_validation_data_rmNoise,
                                                                                    model_input_size)
    test_rm_images, test_rm_labels, test_rm_data_paths = get_images_and_labels_from_filepattern(file_pattern_test_data_rmNoise,
                                                                                       model_input_size)
    val_gauss_images, val_gauss_labels, val_gauss_data_paths = get_images_and_labels_from_filepattern(file_pattern_validation_data_gaussNoise,
                                                                                             model_input_size)
    test_gauss_images, test_gauss_labels, test_gauss_data_paths = get_images_and_labels_from_filepattern(file_pattern_test_data_gaussNoise,
                                                                                                model_input_size)
    batch_size = 64

    train_dataset = RPSDataset(train_images, train_labels, image_transforms)
    val_dataset = RPSDataset(val_images, val_labels, image_transforms)
    test_dataset = RPSDataset(test_images, test_labels, image_transforms)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the saved model
    checkpoint = torch.load('model_states/model_state/baptiste_100epoches__Dropouts_True__BatchNorm_False.pt')
    #checkpoint = torch.load('/home/divingsoup/Documents/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/ModelEng/model_states/model_state/noisy_train_20epoches__Dropouts_True__BatchNorm_False.pt')
    # Define the model architecture
    model = RPS_CNN("final_model_D_True_B_False")
    #model = RPS_CNN("noisy_model_20epochs")
    model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    model.eval()

    # testing
    #print('testing against the training dataset')
    #test_accuracy(model, trainloader)

    val_rm_dataset = RPSDataset(val_rm_images, val_rm_labels, image_transforms)
    test_rm_dataset = RPSDataset(test_rm_images, test_rm_labels, image_transforms)

    val_gauss_dataset = RPSDataset(val_gauss_images, val_gauss_labels, image_transforms)
    test_gauss_dataset = RPSDataset(test_gauss_images, test_gauss_labels, image_transforms)
    
    noisyLoader_val_rm = DataLoader(val_rm_dataset, batch_size=batch_size, shuffle=True)
    noisyLoader_test_rm = DataLoader(test_rm_dataset, batch_size=batch_size, shuffle=True)

    noisyLoader_val_gauss = DataLoader(val_gauss_dataset, batch_size=batch_size, shuffle=True)
    noisyLoader_test_gauss = DataLoader(test_gauss_dataset, batch_size=batch_size, shuffle=True)

    print(f'testing {model.name} against the validation dataset with random noise:')
    test_accuracy(model, noisyLoader_val_rm)

    print(f'testing {model.name} against the validation dataset with gaussian noise:')
    test_accuracy(model, noisyLoader_val_gauss)

    print(f'testing {model.name} against the testing dataset with random noise:')
    test_accuracy(model, noisyLoader_test_rm)

    print(f'testing {model.name} against the testing dataset with gaussian noise:')
    test_accuracy(model, noisyLoader_test_gauss)

    #print('testing against the validation dataset')
    #test_accuracy(model, validationloader)

    print('testing against the testing dataset')
    test_accuracy(model, testloader)

    print_confusion_matrices_new(model, test_rm_dataset, 64)
