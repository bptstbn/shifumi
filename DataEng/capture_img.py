# program to capture single image from webcam in python

# importing OpenCV library
import cv2
import os

# initialize the camera
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that
cam_port = 0
cam = cv2.VideoCapture(cam_port)


def capture_img(img_name = 'test', directory=os.getcwd(), show_in_window=False, save_img=True):
    """
    captures one image from webcam and saves it to directory as png

    :param img_name: name of image without format
    :param directory: where the image should be stored
    :param show_in_window: if True show image in a window
    :param save_img: if true saves the image to the desired directory
    :return:
    """
    # reading the input using the camera
    result, image = cam.read()

    # If image will detected without any error,
    # show result
    if result:

        # showing result, it take frame name and image
        # output
        if show_in_window:
            cv2.imshow(img_name, image)

        # saving image in local storage
        if save_img:
            fullpath = os.path.join(directory, img_name +".png")
            cv2.imwrite(fullpath, image)
            print(f'saved {img_name} to \n\t{fullpath}')

        # If keyboard interrupt occurs, destroy image
        # window
        if show_in_window:
            cv2.waitKey(0)
            cv2.destroyWindow(img_name)

    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")



capture_img( img_name='no_hands', save_img=True, show_in_window=False)