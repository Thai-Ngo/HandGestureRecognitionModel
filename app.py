#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

HAND_LANDMARKS_LEN = 21
HISTORY_LEN = 16
DIMENSION = 2

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Mediapipe drawing utils ##########################################################
    mp_drawing = mp.solutions.drawing_utils

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    general_hand_landmark_history = deque(maxlen=HISTORY_LEN)
    specific_hand_landmark_history = deque(maxlen=HISTORY_LEN)

    # Finger gesture history ################################################
    # finger_gesture_history = deque(maxlen=HISTORY_LEN)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                specific_landmark_list = calc_specific_landmark_list(debug_image, hand_landmarks)
                general_landmark_list = calc_general_landmark_list(hand_landmarks)

                # base_x, base_y = general_landmark_list[0][0], general_landmark_list[0][1]
                # for point in general_landmark_list:
                #     point[0] = point[0] - base_x
                #     point[1] = point[1] - base_y 

                # for point in general_landmark_list:
                #     point[0] = point[0] * debug_image.shape[1] * 0.0000002 + debug_image.shape[1] / 2
                #     point[1] = point[1] * debug_image.shape[0] * 0.0000002 + debug_image.shape[0] / 2

                # for point in general_landmark_list:
                #     cv.circle(debug_image, (int(point[0]), int(point[1])), 5,
                #             (152, 251, 152), 2)
                # print(general_landmark_list)

                # Conversion to relative coordinates / normalized coordinates

                pre_processed_general_landmark_list = pre_process_general_landmark(
                    general_landmark_list)

                pre_processed_general_hand_landmark_history_list = pre_process_general_hand_landmark_history(
                    general_hand_landmark_history
                )

                # Write to the dataset file
                logging_csv(number, mode, pre_processed_general_landmark_list,
                            pre_processed_general_hand_landmark_history_list)

                # Hand sign classification
                # hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
        
                specific_hand_landmark_history.append(specific_landmark_list)
                general_hand_landmark_history.append(general_landmark_list)

                # Finger gesture classification
                # finger_gesture_id = 0
                # point_history_len = len(pre_processed_point_history_list)
                # if point_history_len == (HISTORY_LEN * 2 * 21):
                #     finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                    # if finger_gesture_id == -1:
                    #     print("Not recog")
                    # else:
                    #     print(point_history_classifier_labels[finger_gesture_id])

                # Calculates the gesture IDs in the latest detection
                # finger_gesture_history.append(finger_gesture_id)
                # most_common_fg_id = Counter(
                #     finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(debug_image, brect)
                mp_drawing.draw_landmarks(
                    debug_image,
                    (results.multi_hand_landmarks)[0],
                    mp_hands.HAND_CONNECTIONS
                )
                        
                # debug_image = draw_info_text(
                #     debug_image,
                #     brect,
                #     handedness,
                #     keypoint_classifier_labels[hand_sign_id],
                #     point_history_classifier_labels[most_common_fg_id[0][0]],
                # )
        else:
            specific_hand_landmark_history.append([[0, 0]] * 21)
            general_hand_landmark_history.append([[0, 0]] * 21)

        debug_image = draw_point_history(debug_image, specific_hand_landmark_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_specific_landmark_list(image, landmarks):
    image_width, image_height =  image.shape[1], image.shape[0]
    landmark_points = []
    
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = int(landmark.x * image_width)
        landmark_y = int(landmark.y * image_height)
        # landmark_z = landmark.z
        landmark_points.append([landmark_x, landmark_y])

    # for index, landmark_point in enumerate(landmark_points):
    #     if index == 0:
    #         base_x, base_y = landmark_point[0], landmark_point[1]

    #     landmark_points[index][0] = landmark_points[index][0] - base_x
    #     landmark_points[index][1] = landmark_points[index][1] - base_y

    return landmark_points


def calc_general_landmark_list(landmarks):
    landmark_points = []
    
    # Keypoint
    normalize_factor = 1
    for index, landmark in enumerate(landmarks.landmark):
        if index == 0:
            normalize_factor = abs(landmark.z)

        landmark_x = landmark.x / normalize_factor 
        landmark_y = landmark.y / normalize_factor
        # landmark_z = landmark.z
        landmark_points.append([landmark_x, landmark_y])

    return landmark_points


def pre_process_general_landmark(general_landmark_list):
    temp_general_landmark_list = copy.deepcopy(general_landmark_list)
    for index, landmark_point in enumerate(temp_general_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_general_landmark_list[index][0] = temp_general_landmark_list[index][0] - base_x
        temp_general_landmark_list[index][1] = temp_general_landmark_list[index][1] - base_y

    return [ele for row in temp_general_landmark_list for ele in row]


def pre_process_general_hand_landmark_history(hand_landmark_history):
    temp_hand_landmark_history = copy.deepcopy(hand_landmark_history)

    # Convert to relative coordinates
    base_x, base_y = temp_hand_landmark_history[0][0][0], temp_hand_landmark_history[0][0][1]
    for index, hand in enumerate(temp_hand_landmark_history):
        for point_id in range(len(hand)):
            temp_hand_landmark_history[index][point_id][0] = temp_hand_landmark_history[index][point_id][0] - base_x
            temp_hand_landmark_history[index][point_id][1] = temp_hand_landmark_history[index][point_id][1] - base_y

    # Convert to a one-dimensional list 
    res = []
    for index, hand in enumerate(temp_hand_landmark_history):
        for point_id in range(len(hand)):
            res += [temp_hand_landmark_history[index][point_id][0]]
            res += [temp_hand_landmark_history[index][point_id][1]]
    return res


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0] - 10, brect[1] - 10), (brect[2] + 10, brect[3] + 10),
                     (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0] - 10, brect[1] - 10), (brect[2] + 10, brect[1] - 32),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 14),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
    return image


def draw_point_history(image, hand_history):
    # temp_hand_landmark_history = copy.deepcopy(hand_history)
    # Convert to relative coordinates

    # for index, hand in enumerate(temp_hand_landmark_history):
    #     for point_id in range(len(hand)):
    #         temp_hand_landmark_history[index][point_id][0] = temp_hand_landmark_history[index][point_id][0] * image.shape[1] * 0.0000002
    #         temp_hand_landmark_history[index][point_id][1] = temp_hand_landmark_history[index][point_id][1] * image.shape[0] * 0.0000002

    # base_x, base_y = temp_hand_landmark_history[0][0][0], temp_hand_landmark_history[0][0][1]
    # for index, hand in enumerate(temp_hand_landmark_history):
    #     for point_id in range(len(hand)):
    #         temp_hand_landmark_history[index][point_id][0] = temp_hand_landmark_history[index][point_id][0] - base_x + image.shape[1] / 2
    #         temp_hand_landmark_history[index][point_id][1] = temp_hand_landmark_history[index][point_id][1] - base_y + image.shape[0] / 2

    for index, hand in enumerate(hand_history):
        for point in hand:
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                        (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
