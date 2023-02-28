# methods for data loading

from keras_preprocessing.image import *
import numpy as np
import os
import glob
import random


def get_data(image_paths, target_size=(64, 64), classes=['rock', 'paper', 'scissors']):
    """
    resizes the images to a target size in grayscale

    :param image_paths: list of image paths
    :param target_size: desired image size
    :param classes: the classes which should be predicted
    :return: np.array(images), np.array(labels)
    """
    images, labels = [], []
    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size, color_mode='grayscale')
        image = img_to_array(image)
        label_name = image_path.split(os.path.sep)[-2]
        label = classes.index(label_name)
        label = np.array(label).astype('int32')
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)


def get_images_and_labels_from_filepattern(file_pattern, target_size=(64, 64)):
    """
    reads the images from a filepattern

    :param file_pattern: pattern which specifies which files to read
    :param target_size: output size of images
    :return: images, labels, file_paths
    """
    dataset_paths = [*glob.glob(str(file_pattern))]
    random.shuffle(dataset_paths)
    images, labels = get_data(dataset_paths, target_size=target_size)
    return images, labels.astype('float32'), dataset_paths


from torch.utils.data.dataloader import DataLoader


def show_images_from_dataloader(dataloader: DataLoader):
    """
    Display 8x4 images from a given DataLoader

    :param dataloader: any Dataloader as produced by torch...DataLoader
    """
    data_iter = iter(dataloader)
    sample_images, sample_labels = next(data_iter)
    sample_images, sample_labels = sample_images.cpu().numpy(), sample_labels.cpu().numpy()
    print(f"images.shape: {sample_images.shape} - labels.shape: {sample_labels.shape}")

    # plt.figure(figsize=(5,5))
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        image = sample_images[i]
        # print(f"images[{i}].shape: {image.shape} ")
        image = image.transpose((1, 2, 0))
        # print(f" - AP: images[{i}].shape: {image.shape}")
        plt.imshow(image.squeeze(), cmap='gray')
        #plt.imshow(image.squeeze())
        plt.axis('off')
    plt.show()
    plt.close()


# methods for evaluation

import matplotlib.pyplot as plt
import seaborn as sns

# styling matplotlib
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale=1.10)


def show_validation_loss(history, save_path=None):
    """
    plots the training accuracy from a history as produced by the training in notebook

    :param history: history as produced by training
    :param save_path: he path the image should be saved too
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(range(len(history)), [x['loss'] for x in history], label = 'Training Loss')
    ax.plot(range(len(history)), [x['val_loss'] for x in history], label = 'Validation Loss')
    ax.set(xlabel='epochs', ylabel='loss',
           title='Loss during Training (CrossEntropyLoss)')
    ax.grid()
    # show the labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()


def show_training_accuracy(history, save_path=None, step_size=1):
    """
    plots the training accuracy from a history as produced by the training in notebook

    :param history: history as produced by training
    :param save_path:  the path the image should be saved too
    :param step_size: step size of the x axis
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x_val = [(x+1) * step_size for x in range(len(history))]
    ax.plot(x_val, [x['accuracy_measures'][0]['total'] for x in history], 'r-', label="Total", linewidth=4)
    ax.plot(x_val, [x['accuracy_measures'][0]['rock'] for x in history], 'c-', label="Rock", linewidth=1)
    ax.plot(x_val, [x['accuracy_measures'][0]['paper'] for x in history], 'g-', label="Paper", linewidth=1)
    ax.plot(x_val, [x['accuracy_measures'][0]['scissors'] for x in history], 'y-', label="Scissors",
            linewidth=1)
    ax.set(xlabel='epochs', ylabel='Accuracy in %',
           title='Accuracy in %')
    ax.grid()
    # show the labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()


def show_compare_training_accuracy(histories, names, key='total', save_path=None, colors=['b-', 'g-'], step_size = 1):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for hist, name, color in zip(histories, names, colors):
        x_val = [(x+1) * step_size for x in range(len(hist))]
        ax.plot(x_val, [x['accuracy_measures'][key] for x in hist], color, label=f"{name}",
                linewidth=2)
    ax.set(xlabel='epochs', ylabel='Accuracy in %',
           title=f'Accuracy ({key})  in %')
    ax.grid()
    # show the labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()


def show_compare_training_loss(histories, names, save_path=None, colors=['b-', 'g-'], step_size=1):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for hist, name, color in zip(histories, names, colors):
        x_val = [(x + 1) * step_size for x in range(len(hist))]
        ax.plot(x_val, [x['loss'] for x in hist], color, label=f"{name}",
                linewidth=2)
    ax.set(xlabel='epochs', ylabel='loss',
           title='Loss during Training (CrossEntropyLoss)')
    ax.grid()
    # show the labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()

def convert_seconds(seconds):
    """
    converts seconds into hours minutes and seconds
    :param seconds: the seconds you want to convert e.g. from time differences
    :return: string of hh:mm:ss
    """
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return f"{int(hours)}:{int(minutes)}:{int(seconds)}"


# save history to a file and reload it
import pickle


def save_history(save_path, object):
    """
    uses pickle to dub bytes of an object

    :param save_path: where to save the object
    :param object: any python object to be saved
    """
    with open(save_path, "wb") as fp:  # Pickling
        pickle.dump(object, fp)


def load_history(load_path):
    """
    uses pickle load to load any pickle dumped object

    :param load_path: path of pickle dumped object
    :return: loaded object using pickle load
    """
    with open(load_path, "rb") as fp:  # Unpickling
        return pickle.load(fp)


"""
# set file path as working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))



h1 = load_history(os.getcwd() + '\model_states\history__baptiste_100epoches__Dropouts_False__BatchNorm_False')
h2 = load_history(os.getcwd() + '\model_states\history__baptiste_100epoches__Dropouts_True__BatchNorm_False')
h3 = load_history(os.getcwd() + '\model_states\history__baptiste_100epoches__Dropouts_False__BatchNorm_True')
h4 = load_history(os.getcwd() + '\model_states\history__baptiste_100epoches__Dropouts_True__BatchNorm_True')


histories = [h1, h2, h3, h4]
names = ['No regularization', 'Dropout', 'BatchNorm', 'Dropout & BatchNorm']

savepath = f'{os.getcwd()}\model_states\obaptiste_val_accuracies_comparison'

show_compare_training_accuracy(histories, names, key='total', save_path=savepath, colors=['b-', 'g-', 'r-', 'black'], step_size = 1)

"""

def show_training_and_val_accuracy(history, save_path=None, step_size=1):
    """
    plots the training accuracy from a history as produced by the training in notebook

    :param history: history as produced by training
    :param save_path:  the path the image should be saved too
    :param step_size: step size of the x axis
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x_val = [(x+1) * step_size for x in range(len(history))]
    ax.plot(x_val, [x['training_accuracy'] for x in history], 'b-', label="Training Accuracy", linewidth=1)
    ax.plot(x_val, [x['accuracy_measures']['total'] for x in history], 'g-', label="Validation Accuracy", linewidth=1)
    ax.set(xlabel='epochs', ylabel='Accuracy in %',
           title='Accuracy in %')
    ax.grid()
    # show the labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()




"""
for dropout in [True, False]:
    for batchnorm in [True, False]:
        h = load_history(os.getcwd() + '\model_states\history__baptiste_100epoches__Dropouts_' + str(dropout) + '__BatchNorm_' + str(batchnorm))
        savepath = f'{os.getcwd()}\model_states\obaptiste_100epoches_train_accuracy__Dropouts_{str(dropout)}__BatchNorm_{str(batchnorm)}'
        show_training_and_val_accuracy(h, savepath)
"""
                               
                               