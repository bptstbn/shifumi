"""
This script is written by Jonas Amling

I will try to use mediapipe to detect hands since they do have a hand detection model
I will use rembg to remove the background of an image
I will use cv2 to load the image and resize it to the desired size
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove


class Preprocessor():
    def __init__(self, remove_background: bool = True, dim_x: int = 200, dim_y: int = 200):
        """
        constructs the Preprocessor

        :param remove_background: if true, image background is removed
        :param dim_x: number of pixels in x dimension
        :param dim_y: number of pixels in y dimension
        """
        self.remove_background = remove_background
        self.dimensions = (dim_x, dim_y)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    def __call__(self, image_path: str):
        """
        Takes the path to an image and returns the preprocessed version of the image

        :param image_path: path to the image that should be processed
        :return: the preprocessed image
        """
        # import the image
        image = cv2.flip(cv2.imread(image_path), 1)
        # crop image using media pipe hand detection
        image = self.__crop_image(image)
        # remove the background
        if self.remove_background:
            image = self.__remove_bg(image)
        # TODO bring the image into the desired format
        # use cv2.resize
        # here you should return the preprocessed image
        return cv2.flip(image, 1)

    def __resize(self, image):
        # important, keep the aspect ratio
        # first resize so that width or heigth fits the newly desired ratio, depending on what is bigger
        # fill either height or width with additional rows until desired size is reached
        # fill missing outer pixels
        raise NotImplementedError("not implemented yet")

    def __remove_bg(self, image):
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
            # TODO check if a hand is found
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
        bbox = np.min(np_x_hand_cords), \
               np.max(np_x_hand_cords), \
               np.min(np_y_hand_cords), \
               np.max(np_y_hand_cords)
        # calculate the offset
        # move x and y points according to min and max differences and offset desired
        offest = offest / 100
        bbox = bbox[0] - (bbox[1] - bbox[0]) * offest, \
               bbox[1] + (bbox[1] - bbox[0]) * offest, \
               bbox[2] - (bbox[3] - bbox[2]) * offest, \
               bbox[3] + (bbox[3] - bbox[2]) * offest

        return bbox

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
        for file in os.listdir(input_directory):
            if file.endswith(tuple(allowed_file_endings)):
                # save the preprocessed file to the new folder
                try:
                    self.save_image(self(os.path.join(input_directory, file)), output_directory, file.split('.')[0])
                except Exception as e:
                    print(f'not able to process: \n\t{os.path.join(input_directory, file)}\n\n{e}')


if __name__ == "__main__":
    img_path = '/Users/amling/uni/shifumi/DataEng/no_hands.png'
    test_processor = Preprocessor(remove_background=True)
    test_processor.preprocess_entire_folder('test_images', 'test_images_out')
