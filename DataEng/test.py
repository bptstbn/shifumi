import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
# TODO throw in an image and detect the hand
IMAGE_FILES = [os.path.join('test_images','2PAcPusQ59xIMfiw.png')]


def find_bounding_box(results, image_width, image_height, offest):
    """
    finds the 4 corners of the bounding box

    :param hand_landmarks: landmarks of a hand
    :param offest: offset in %
    :return: min_x, max_x, min_y, max_y
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
    offest = offest/100
    bbox= bbox[0] - (bbox[1] - bbox[0]) * offest, \
          bbox[1] + (bbox[1] - bbox[0]) * offest, \
          bbox[2] - (bbox[3] - bbox[2]) * offest, \
          bbox[3] + (bbox[3] - bbox[2]) * offest

    return bbox


with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        # print('Handedness:', results.multi_handedness)
        # if not results.multi_hand_landmarks:
        #     continue
        image_height, image_width, _ = image.shape
        print(f' image height= {image_height} , image width= {image_width}')
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        bounding_box = find_bounding_box(results, image_width, image_height, offest=10)
        print(f'bounding box = {bounding_box}')
        start_pint=(int(bounding_box[0]), int(bounding_box[2]))
        stop_pint=(int(bounding_box[1]), int(bounding_box[3]))

        annotated_image = cv2.rectangle(annotated_image, start_pint, stop_pint, (255,0,0))
        cropped_image = image.copy()
        cropped_image= cropped_image[start_pint[1]:stop_pint[1], start_pint[0]:stop_pint[0]]
        cv2.imwrite(
            '/Users/amling/uni/shifumi/DataEng/test_annotated' + str(idx) + '.png', cv2.flip(annotated_image, 1))
        cv2.imwrite(
            '/Users/amling/uni/shifumi/DataEng/test_cropped' + str(idx) + '.png', cv2.flip(cropped_image, 1))
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            continue
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
