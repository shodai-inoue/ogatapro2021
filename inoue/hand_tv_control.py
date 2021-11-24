#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import serial

from utils import CvFpsCalc

from enum import Enum

ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)

"""test"""
class Pose(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    NONE = -1

HANDPOSE_JUDGE = {
    (False, False, False, False, False): Pose.ZERO,
    (False, True, False, False, False): Pose.ONE,
    (False, True, True, False, False): Pose.TWO,
    (False, True, True, True, False): Pose.THREE,
    (False, True, True, True, True): Pose.FOUR,
    (True, False, False, False, False): Pose.FIVE,
    (True, True, False, False, False): Pose.SIX,
    (True, True, True, False, False): Pose.SEVEN,
    (True, True, True, True, False): Pose.EIGHT,
    (True, True, True, True, True): Pose.NINE,
}

thumbIsOpen = False; # 親指
firstFingerIsOpen = False; # 人差し指
secondFingerIsOpen = False; # 中指
thirdFingerIsOpen = False; # 薬指
fourthFingerIsOpen = False; # 小指

def judge_handpose(fingerIsOpenTuple):
    if not fingerIsOpenTuple in HANDPOSE_JUDGE:
        return Pose.NONE
    else:
        return HANDPOSE_JUDGE[fingerIsOpenTuple]

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args

count = 0
def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_hands = 1
    # max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(1)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640) # 160
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480) # 120

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # 指の状態 ########################################################
    global thumbIsOpen, firstFingerIsOpen, secondFingerIsOpen, thirdFingerIsOpen, fourthFingerIsOpen, count
    # thumbIsOpen = False; # 親指
    # firstFingerIsOpen = False; # 人差し指
    # secondFingerIsOpen = False; # 中指
    # thirdFingerIsOpen = False; # 薬指
    # fourthFingerIsOpen = False; # 小指

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        # 描画 ################################################################
        # finger_state = None
        # hand_pose = None
        thumb1 = thumb2 = thumb3 = first1 = first2 = first3 = second1 = second2 = second3 = third1 = third2 = third3 = fourth1 = fourth2 = fourth3 = (0,0)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, hand_landmarks)
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # 描画
                debug_image, thumb1, thumb2, thumb3, first1, first2, first3, second1, second2, second3, third1, third2, third3, fourth1, fourth2, fourth3 = draw_landmarks(debug_image, cx, cy, hand_landmarks, handedness)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        pseudoFixKeyPoint = thumb1[0]
        if thumb2[0] < pseudoFixKeyPoint and thumb3[0] < pseudoFixKeyPoint:
            thumbIsOpen = True
        else:
            thumbIsOpen = False

        pseudoFixKeyPoint = first1[1]
        if first2[1] < pseudoFixKeyPoint and first3[1] < pseudoFixKeyPoint:
            firstFingerIsOpen = True
        else:
            firstFingerIsOpen = False

        pseudoFixKeyPoint = second1[1]
        if second2[1] < pseudoFixKeyPoint and second3[1] < pseudoFixKeyPoint:
            secondFingerIsOpen = True
        else:
            secondFingerIsOpen = False

        pseudoFixKeyPoint = third1[1]
        if third2[1] < pseudoFixKeyPoint and third3[1] < pseudoFixKeyPoint:
            thirdFingerIsOpen = True
        else:
            thirdFingerIsOpen = False

        pseudoFixKeyPoint = fourth1[1]
        if fourth2[1] < pseudoFixKeyPoint and fourth3[1] < pseudoFixKeyPoint:
            fourthFingerIsOpen = True
        else:
            fourthFingerIsOpen = False

        finger_state = (thumbIsOpen, firstFingerIsOpen, secondFingerIsOpen, thirdFingerIsOpen, fourthFingerIsOpen)

        hand_pose = judge_handpose(finger_state)

        if hand_pose == Pose.ONE: # ２秒間同じポーズだったら
            count += 1
            if count > 10:
                ser.write("channel:1".encode())
                time.sleep(1)
                count = 0

        elif hand_pose == Pose.TWO:
            count += 1
            if count > 10:
                ser.write("channel:2".encode())
                time.sleep(1)
                count = 0
           

        elif hand_pose == Pose.THREE:
            count += 1
            if count > 10:
                ser.write("channel:3".encode())
                time.sleep(1)
                count = 0

        elif hand_pose == Pose.FOUR:
            count += 1
            if count > 10:
                ser.write("channel:4".encode())
                time.sleep(1)
                count = 0

        elif hand_pose == Pose.FIVE:
            count += 1
            if count > 10:
                ser.write("channel:5".encode())
                time.sleep(1)
                count = 0

        elif hand_pose == Pose.SIX:
            count += 1
            if count > 10:
                ser.write("channel:6".encode())
                time.sleep(1)
                count = 0

        elif hand_pose == Pose.SEVEN:
            count += 1
            if count > 10:
                ser.write("channel:7".encode())
                time.sleep(1)
                count = 0

        elif hand_pose == Pose.EIGHT:
            count += 1
            if count > 10:
                ser.write("channel:8".encode())
                time.sleep(1)
                count = 0

        elif hand_pose == Pose.NINE:
            count += 1
            if count > 10:
                ser.write("channel:9".encode())
                time.sleep(1)
                count = 0

            # hand_landmarks = results.multi_hand_landmarks
            # Get coordinates
            # pseudoFixKeyPoint = hand_landmarks[2].x
            # if hand_landmarks[3].x < pseudoFixKeyPoint and hand_landmarks[4].x < pseudoFixKeyPoint.x:
            #     thumbIsOpen = True

            # pseudoFixKeyPoint = hand_landmarks[6].y
            # if hand_landmarks[7].y < pseudoFixKeyPoint and hand_landmarks[8].y < pseudoFixKeyPoint:
            #     firstFingerIsOpen = True

            # pseudoFixKeyPoint = hand_landmarks[10].y
            # if hand_landmarks[11].y < pseudoFixKeyPoint and hand_landmarks[12].y < pseudoFixKeyPoint:
            #     secondFingerIsOpen = True

            # pseudoFixKeyPoint = hand_landmarks[14].y
            # if hand_landmarks[15].y < pseudoFixKeyPoint and hand_landmarks[16].y < pseudoFixKeyPoint:
            #     thirdFingerIsOpen = True

            # pseudoFixKeyPoint = hand_landmarks[18].y
            # if hand_landmarks[19].y < pseudoFixKeyPoint and hand_landmarks[20].y < pseudoFixKeyPoint:
            #     fourthFingerIsOpen = True

        
                
                # Visualize angle
        cv.putText(debug_image, str(hand_pose), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
        #            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MediaPipe Hand Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()

# def finger_state_func(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]

#     palm_array = np.empty((0, 2), int)

#     for index, landmark in enumerate(landmarks.landmark):
#         x = min(int(landmark.x * image_width), image_width - 1)
#         y = min(int(landmark.y * image_height), image_height - 1)

#         landmark_point = [np.array((x, y))]

#         # Get coordinates
#         pseudoFixKeyPoint = landmarks[2].x
#         if landmarks[3].x < pseudoFixKeyPoint and landmarks[4].x < pseudoFixKeyPoint:
#             thumbIsOpen = True

#         pseudoFixKeyPoint = landmarks[6].y
#         if landmarks[7].y < pseudoFixKeyPoint and landmarks[8].y < pseudoFixKeyPoint:
#             firstFingerIsOpen = True

#         pseudoFixKeyPoint = landmarks[10].y
#         if landmarks[11].y < pseudoFixKeyPoint and landmarks[12].y < pseudoFixKeyPoint:
#             secondFingerIsOpen = True

#         pseudoFixKeyPoint = landmarks[14].y
#         if landmarks[15].y < pseudoFixKeyPoint and landmarks[16].y < pseudoFixKeyPoint:
#             thirdFingerIsOpen = True

#         pseudoFixKeyPoint = landmarks[18].y
#         if landmarks[19].y < pseudoFixKeyPoint and landmarks[20].y < pseudoFixKeyPoint:
#             fourthFingerIsOpen = True



def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


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


def draw_landmarks(image, cx, cy, landmarks, handedness):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手首1
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手首2
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            thumb1 = (landmark_x, landmark_y)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            thumb2 = (landmark_x, landmark_y)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            thumb3 = (landmark_x, landmark_y)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            first1 = (landmark_x, landmark_y)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            first2 = (landmark_x, landmark_y)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            first3 = (landmark_x, landmark_y)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            second1 = (landmark_x, landmark_y)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            second2 = (landmark_x, landmark_y)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            second3 = (landmark_x, landmark_y)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            third1 = (landmark_x, landmark_y)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            third2 = (landmark_x, landmark_y)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            third3 = (landmark_x, landmark_y)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            fourth1 = (landmark_x, landmark_y)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            fourth2 = (landmark_x, landmark_y)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            fourth3 = (landmark_x, landmark_y)

    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

        # 人差指
        cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

        # 中指
        cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

        # 薬指
        cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

        # 小指
        cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

        # 手の平
        cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        # handedness.classification[0].index
        # handedness.classification[0].score

        cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv.putText(image, handedness.classification[0].label[0], (cx - 6, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)  # label[0]:一文字目だけ

    return image, thumb1, thumb2, thumb3, first1, first2, first3, second1, second2, second3, third1, third2, third3, fourth1, fourth2, fourth3


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    main()
