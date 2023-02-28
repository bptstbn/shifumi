import os
from notebook_utils import *
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from sklearn.metrics import confusion_matrix

# set file path as working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

CLASSES = ['rock', 'paper', 'scissors']
model_input_size = (64, 64)

file_pattern_training_data = 'C:/Users/bmars/Desktop/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/combined/*/*.png'
file_pattern_validation_data = 'C:/Users/bmars/Desktop/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/validation_grey_pp01/*/*.png'
file_pattern_test_data = 'C:/Users/bmars/Desktop/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/testing_grey_pp01/*/*.png'

file_pattern_validation_data_rmNoise = 'C:/Users/bmars/Desktop/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/validation_grey_pp01_randomNoise/*/*.png'
file_pattern_validation_data_gaussNoise = 'C:/Users/bmars/Desktop/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/validation_grey_pp01_gaussianNoise/*/*.png'

file_pattern_test_data_rmNoise = 'C:/Users/bmars/Desktop/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/testing_grey_pp01_rm_noise/*/*.png'
file_pattern_test_data_gaussNoise = 'C:/Users/bmars/Desktop/Uni-Bamberg/Semester5/ProjectXDeepLearning/shifumi/DataEng/datasets/testing_grey_pp01_gaussian_noise/*/*.png'


# unchanged model architecture to train with noisy data
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


# old incorrect evaluation method
def test_accuracy(model, testloader, criterion=None):
    # Test the model on the test dataset
    # print("test accuracy called")
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
        # print(f'Loss of the network on the test set: {running_loss/len(testloader)}')
        outcome['total'] = 100 * correct / total
        for i in range(len(CLASSES)):
            if class_total[1] > 0:
                print(f'Accuracy of {CLASSES[i]} : {100 * class_correct[i] / class_total[i]}%')
                outcome[CLASSES[i]] = 100 * class_correct[i] / class_total[i]
            else:
                print(f'Accuracy of {CLASSES[i]} : could not be measured, no test images%')
    return outcome, running_loss / len(testloader)


# unchanged training method
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
        hist.append({'loss': epoch_loss, 'val_loss': val_loss, 'accuracy_measures': val_outcome,
                     'training_accuracy': training_accuracy})
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


class RandomPixelRemoval(object):
    """
    A callable class that randomly removes a specified percentage of pixels from an image.

    :param p: Probability of applying the transformation (default 0.3).
    :type p: float
    :param remove_pct: Percentage of pixels to remove from the image (default 0.25).
    :type remove_pct: float

    :return: Transformed image.
    :rtype: numpy.ndarray
    """

    def __init__(self, p=0.3, remove_pct=0.25):
        self.p = p
        self.remove_pct = remove_pct

    def __call__(self, image):
        """
        Applies the random pixel removal transformation to the given image.

        :param image: The input image to transform.
        :type image: numpy.ndarray

        :return: The transformed image.
        :rtype: numpy.ndarray
        """

        if random.uniform(0, 1) > self.p:
            return image

        h, w, _ = image.shape
        num_pixels_to_remove = int(h * w * self.remove_pct)

        # Remove random pixels
        for i in range(num_pixels_to_remove):
            row = random.randint(0, h - 1)
            col = random.randint(0, w - 1)
            image[row, col] = 0

        return image


def remove_parts_transformation(image, prob=0.25):
    """
    Removes parts of the image based on a randomly generated mask.

    :param image: The input image to transform.
    :type image: numpy.ndarray
    :param prob: Probability of removing each pixel in the mask (default 0.25).
    :type prob: float

    :return: The transformed image.
    :rtype: torch.Tensor
    """

    # Generate a random mask
    mask = np.random.rand(image.shape[0], image.shape[1]) < prob
    mask = torch.from_numpy(mask)
    image[mask] = torch.tensor([0, 0, 0], dtype=torch.float32)
    return image


class WithNoiseTransform(object):
    """
    A callable class that applies a series of random image transformations, including random pixel removal and Gaussian blur.

    :param normal_prob: Probability of not applying the pixel removal transformation (default 0.7).
    :type normal_prob: float

    :return: The transformed image.
    :rtype: numpy.ndarray
    """

    def __init__(self, normal_prob=0.7):
        self.normal_prob = normal_prob

    def __call__(self, image):
        """
        Applies the series of random image transformations to the given image.

        :param image: The input image to transform.
        :type image: numpy.ndarray

        :return: The transformed image.
        :rtype: numpy.ndarray
        """

        if random.random() > self.normal_prob:
            return remove_parts_transformation(image)
        else:
            return image


# Original Transformation for Dataloader
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(0, shear=0.2),
    transforms.RandomAffine(0, scale=(0.8, 1.2)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

# Transformation used for testing (no augmentations)
image_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Transformation for Dataloader that uses 30% randomly distorted data
train_withNoise_transforms = transforms.Compose([
    RandomPixelRemoval(),
    transforms.ToTensor(),
    transforms.RandomAffine(0, shear=0.2),
    transforms.RandomAffine(0, scale=(0.8, 1.2)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

# Transformation for Dataloader that uses Gaussian Noise
train_withNoise_transforms2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=1, sigma=25),
    transforms.RandomAffine(0, shear=0.2),
    transforms.RandomAffine(0, scale=(0.8, 1.2)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])


def get_all_predictions(model, loader):
    """
    Computes the predictions of a given model on a given data loader.

    :param model: PyTorch model.
    :param loader: DataLoader object.
    :return: Tensor containing all the model's predictions.
    """

    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            , dim=0
        )
    return all_preds


def mcc(tp, fp, tn, fn):
    """
    Computes the Matthews correlation coefficient (MCC) given true positive (tp),
    false positive (fp), true negative (tn), and false negative (fn) counts.

    :param tp: int, number of true positives.
    :param fp: int, number of false positives.
    :param tn: int, number of true negatives.
    :param fn: int, number of false negatives.
    :return: float, the computed MCC score.
    """
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_score = numerator / denominator
    return mcc_score


def get_eval_metrics(model, dataset, batch_size):
    """
      Computes evaluation metrics for a model on a given dataset.

      :param model: PyTorch model.
      :param dataset: PyTorch dataset.
      :param batch_size: int, batch size for evaluation.
      :return: tuple, containing confusion matrix, precision, recall, F1 score,
          accuracy, overall accuracy, and Matthews correlation coefficient.
      """

    with torch.no_grad():
        total_samples = len(dataset)
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        for i in range(0, total_samples, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, total_samples)
            batch_dataset = torch.utils.data.Subset(dataset, range(batch_start, batch_end))
            prediction_loader = DataLoader(batch_dataset, batch_size=batch_size)
            train_preds = get_all_predictions(model, prediction_loader)

            truth_loader = DataLoader(batch_dataset, batch_size=batch_size, shuffle=False)

            test_targets_list = []
            for inputs, targets in truth_loader:
                test_targets_list.append(targets)

            test_targets = test_targets_list[0]

            all_preds = torch.cat((all_preds, train_preds.argmax(dim=1)))
            all_labels = torch.cat((all_labels, test_targets))

    cm = confusion_matrix(all_labels, all_preds)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.sum(cm) - tp - fp - fn
    mcc_score = mcc(tp=tp, fp=fp, tn=tn, fn=fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    overall_accuracy = np.sum(np.diag(cm))/np.sum(cm)

    return cm, precision, recall, f1score, accuracy, overall_accuracy, mcc_score


def print_metrics_for_individual_classes(model, dataset, batch_size):
    """
    Prints the precision, recall, F1-score and accuracy for each individual class, as well as the overall accuracy and
    Matthews correlation coefficient (MCC) for a given PyTorch model and dataset.

    :param model: PyTorch model used for prediction.
    :param dataset: PyTorch dataset object used for evaluation.
    :param batch_size: int, batch size used for evaluation.
    """

    cm, precision, recall, f1score, accuracy, overall_accuracy, mcc_score = get_eval_metrics(model, dataset, batch_size)
    labels = ['rock', 'paper', 'scissors']
    print(cm)
    for i, label in enumerate(labels):
        print(
            f'{label}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f},'
            f' F1-score={f1score[i]:.3f}, Accuracy={accuracy[i]:.3f}')
    print(f' Total Accuracy: {overall_accuracy}, MCC: {mcc_score}')


def print_metrics_and_cm(model, dataset, batch_size):
    """
    Computes and prints the evaluation metrics and displays the confusion matrix in toatal numbers and in percent using Matplotlib.
    Also,

    :param model: PyTorch model used for prediction.
    :param dataset: PyTorch dataset object used for evaluation.
    :param batch_size: int, batch size used for evaluation.
    """

    cm, precision, recall, f1score, accuracy, overall_accuracy, mcc_score = get_eval_metrics(model, dataset, batch_size)
    labels = ['rock', 'paper', 'scissors']
    print(cm)
    for i, label in enumerate(labels):
        print(
            f'{label}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f},'
            f' F1-score={f1score[i]:.3f}, Accuracy={accuracy[i]:.3f}')
    print(f' Total Accuracy: {overall_accuracy}, MCC: {mcc_score}')

    # Print the confusion matrix using Matplotlib
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.0f')

    ax.set_title('RPS Classification\n\n')
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual')

    ## Ticket labels
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

    ## Ticket labels
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
    val_rm_images, val_rm_labels, val_rm_data_paths = get_images_and_labels_from_filepattern(
        file_pattern_validation_data_rmNoise,
        model_input_size)
    test_rm_images, test_rm_labels, test_rm_data_paths = get_images_and_labels_from_filepattern(
        file_pattern_test_data_rmNoise,
        model_input_size)
    val_gauss_images, val_gauss_labels, val_gauss_data_paths = get_images_and_labels_from_filepattern(
        file_pattern_validation_data_gaussNoise,
        model_input_size)
    test_gauss_images, test_gauss_labels, test_gauss_data_paths = get_images_and_labels_from_filepattern(
        file_pattern_test_data_gaussNoise,
        model_input_size)
    batch_size = 64

    train_dataset = RPSDataset(train_images, train_labels, image_transforms)
    val_dataset = RPSDataset(val_images, val_labels, image_transforms)
    test_dataset = RPSDataset(test_images, test_labels, image_transforms)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the saved model
    # checkpoint = torch.load('model_states/model_state/baptiste_100epoches__Dropouts_True__BatchNorm_False.pt')
    checkpoint = torch.load('model_states/model_state/noisy_train_20epoches__Dropouts_True__BatchNorm_False.pt')

    # Define the model architecture
    # model = RPS_CNN("final_model_D_True_B_False")
    model = RPS_CNN("noisy_model_20epochs")
    model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    model.eval()

    # Create datasets
    val_rm_dataset = RPSDataset(val_rm_images, val_rm_labels, image_transforms)
    test_rm_dataset = RPSDataset(test_rm_images, test_rm_labels, image_transforms)

    val_gauss_dataset = RPSDataset(val_gauss_images, val_gauss_labels, image_transforms)
    test_gauss_dataset = RPSDataset(test_gauss_images, test_gauss_labels, image_transforms)

    #print(f'testing {model.name} against the training dataset:')
    #print_metrics_and_cm(model, train_dataset, 64)

    #print(f'testing {model.name} against the validation dataset with random noise:')
    #print_metrics_and_cm(model, val_rm_dataset, 64)

    #print(f'testing {model.name} against the validation dataset with gaussian noise:')
    #print_metrics_and_cm(model, val_gauss_dataset, 64)

    #print(f'testing {model.name} against the testing dataset with random noise:')
    #print_metrics_and_cm(model, test_rm_dataset, 64)

    print(f'testing {model.name} against the testing dataset with gaussian noise:')
    print_metrics_and_cm(model, test_gauss_dataset, 64)


