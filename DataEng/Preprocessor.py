"""
This script is written by Jonas Amling

I will try to use mediapipe to detect hands since they do have a hand detection model
I will use rembg to remove the background of an image
I will use cv2 to load the image and resize it to the desired size
"""

import os
from typing import Tuple

import cv2
import numpy as np
import mediapipe as mp
from rembg import remove


class Preprocessor():
    def __init__(self, remove_background: bool = True, crop_image: bool = True, dim_x: int = 200, dim_y: int = 200, greyscale: bool = True):
        """
        constructs the Preprocessor

        :param remove_background: if true, image background is removed
        :param crop_image: wether or not to crop the image based on te mediapipe hand model
        :param dim_x: number of pixels in x dimension
        :param dim_y: number of pixels in y dimension
        :param greyscale: if set to True images are preprocessed into greyscale images
        """
        self.remove_background = remove_background
        self.crop_image = crop_image
        self.desired_dimensions = (dim_y, dim_x)
        self.mp_hands = mp.solutions.hands
        self.greyscale = greyscale

    def __call__(self, image_path: str):
        """
        Takes the path to an image and returns the preprocessed version of the image

        :param image_path: path to the image that should be processed
        :return: the preprocessed image
        """
        # import the image, and flip since this is needed for mediapipe
        image = cv2.flip(cv2.imread(image_path), 1)
        # crop image using media pipe hand detection
        if self.crop_image:
            image = self.__crop_image(image)
        # remove the background
        if self.remove_background:
            image = self.__remove_bg(image)
        if self.greyscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.__resize(image, self.desired_dimensions)
        # here you should return the preprocessed image
        return cv2.flip(image, 1)

    def __resize(self,
                 image: np.array,
                 new_shape: Tuple[int, int],
                 padding_color: Tuple[int] = (0, 0, 0)):
        """
        Maintains aspect ratio and resizes with padding.

        see https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec

        :param image: Image to be resized.
        :param new_shape: Expected (width, height) of new image.
        :param padding_color: Tuple in BGR of padding color
        :return: Resized image with padding
        """
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        return image

    def __remove_bg(self, image):
        """
        simply removes background using the rembg library

        :param image: image, cv2.imread
        :return: image, with transparent background
        """
        return remove(image)

    def __crop_image(self, image):
        """
        crops the image according to the hand found by mediapipe

        :param image: cv2 image, flipped
        :return: cropped image if hand is found, otherwise same image
        """
        with self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5) as hands:
            # detect the hand using mediapipe
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # checks if hand is found
            if results.multi_hand_landmarks:
                image_height, image_width, _ = image.shape
                bounding_box = self.__find_bounding_box(results, image_width, image_height, offest=10)
                start_point = (int(bounding_box[0]), int(bounding_box[2]))
                stop_point = (int(bounding_box[1]), int(bounding_box[3]))
                cropped_image = image.copy()
                return cropped_image[start_point[1]:stop_point[1], start_point[0]:stop_point[0]]
            else:
                return image

    def __find_bounding_box(self, results, image_width, image_height, offest):
        """
        finds the bounding box according to the mediapipe ands result

        :param results:
        :param image_width:
        :param image_height:
        :param offest:
        :return:
        """
        # first map landmaks to (x.y tuples)
        x_hand_cords = []
        y_hand_cords = []
        # iterate over the 21 hand landmarks found my mediapipe
        for hand_landmarks in results.multi_hand_landmarks:
            for x in range(21):
                x_cord = hand_landmarks.landmark[x].x * image_width
                y_cord = hand_landmarks.landmark[x].y * image_height
                x_hand_cords.append(x_cord)
                y_hand_cords.append(y_cord)
        # convert to numpy arrays and find min and maximum x and y coordinates -> the bounding box
        np_x_hand_cords = np.asarray(x_hand_cords)
        np_y_hand_cords = np.asarray(y_hand_cords)
        min_x = np.min(np_x_hand_cords)
        min_y = np.max(np_x_hand_cords)
        max_x = np.min(np_y_hand_cords)
        max_y = np.max(np_y_hand_cords)
        # calculate the offset
        # move x and y points according to min and max differences and offset desired
        offest = offest / 100
        min_x = min_x - (min_y - min_x) * offest
        min_y = min_y + (min_y - min_x) * offest
        max_x = max_x - (max_y - max_x) * offest
        max_y = max_y + (max_y - max_x) * offest
        # set min x and min y to minimum of 0
        if min_x < 0:  # min x
            min_x = 0
        if max_x < 0:  # min y
            max_x = 0
        # set max x and max y to maximum of image_width or height
        if min_y > image_width:  # max x
            min_y = image_width
        if max_y > image_height:  # max y
            max_y = image_height

        return min_x, min_y, max_x, max_y

    def save_image(self, image, directory, name):
        """
        saves a preprocessed image to specified directory

        :param image: preprocessed image
        :param directory: the directory where it should be stored
        :param name: some_name
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        # TODO make sure this works on windows
        cv2.imwrite(os.path.join(directory, f'{name}.png'), image)

    def preprocess_entire_folder(self, input_directory, output_directory, allowed_file_endings=['.png']):
        print(f'process all files ind {input_directory}')
        for file in os.listdir(input_directory):
            if file.endswith(tuple(allowed_file_endings)):
                # save the preprocessed file to the new folder
                try:
                    self.save_image(self(os.path.join(input_directory, file)), output_directory, file.split('.')[0])
                    print(f'\tprocessed: {file} and written into {output_directory}')
                except Exception as e:
                    print(f'not able to process: \n\t{os.path.join(input_directory, file)}\n\n{e}')

    def preprocess_entire_dataset(self, input_dir: str, output_dir: str, allowed_file_endings=['.png']):
        """
        preprocesses an entire dataset recursively using self.preprocess_entire_folder

        :param input_dir: directory of the dataset
        :param output_dir:  output directory where dataset should be stored
        :param allowed_file_endings: list of allowed file endings
        :return:
        """
        self.preprocess_entire_folder(input_dir, output_dir, allowed_file_endings)
        # For each folder call recursively
        subdirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]
        for s in subdirs:
            self.preprocess_entire_dataset(s,
                                           os.path.join(output_dir, s.split('/')[-1]),
                                           allowed_file_endings)


if __name__ == "__main__":
    img_path = '/Users/amling/uni/shifumi/DataEng/no_hands.png'
    test_processor = Preprocessor(remove_background=False, greyscale=True)
    # test_processor.preprocess_entire_folder('test_images', 'test_images_out')
    # test_processor.save_image(test_processor(os.path.join('test_images', '2PAcPusQ59xIMfiw.png')), 'test_images_out_wo',
    #                          '2PAcPusQ59xIMfiw')
    dataset = os.path.join('datasets', 'jonas')
    out = os.path.join('datasets', 'jonas_pp')
    test_processor.preprocess_entire_dataset(dataset, out)
