import argparse
import logging
import time
from pprint import pprint
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
# import RPi.GPIO as GPIO
import serial

# Pin Definitons:
# led_pin = 18  # Board pin 12
# but_pin = 18  # Board pin 18
ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

# https://github.com/JetsonHacksNano/CSI-Camera
# https://forums.developer.nvidia.com/t/how-to-eliminate-gstreamer-camera-buffer/75608/9
def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=2) :
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True' % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

# GST_STR = 'nvarguscamerasrc \
#     ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 \
#     ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx \
#     ! videoconvert \
#     ! appsink'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def find_point(pose, p):
    for point in pose:
        try:
            body_part = point.body_parts[p]
            return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
        except:
            return (0,0)
    return (0,0)

def euclidian( point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 )

def angle_calc(p0, p1, p2 ):
    
    try:
        a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
        angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi
    except:
        return 0
    return int(angle)

def muscle_pose( a, b, c, d):
    
    if a in range(140,180) and b in range(55,95) and c in range(140,180) and d in range(55,95):
        return True
    return False

def fold_arms( a, b, c, d):
    
    if a in range(90,115) and b in range(65,90) and c in range(90,115) and d in range(65,90):
        return True
    return False

def running_man_left( a, b, c, d):
    
    if a in range(75,100) and b in range(65,90) and c in range(165,180) and d in range(165,180):
        return True
    return False

def running_man_right( a, b, c, d):
    
    if a in range(165,180) and b in range(165,180) and c in range(75,100) and d in range(65,90):
        return True
    return False

def lhand_on_relbow( a, b, c, d):
    
    if a in range(100,120) and b in range(10,30) and c in range(90,115) and d in range(65,90):
        return True
    return False

def rhand_on_lelbow( a, b, c, d):
    
    if a in range(90,115) and b in range(65,90) and c in range(100,120) and d in range(10,30):
        return True
    return False

def hand_on_hip( a, b, c, d):
    
    if a in range(110,130) and b in range(115,135) and c in range(110,130) and d in range(115,135):
        return True
    return False

def hand_on_chin( a, b, c, d):
    
    if a in range(75,95) and b in range(75,95) and c in range(85,105) and d in range(15,30):
        return True
    return False

def draw_str(dst, xxx_todo_changeme, s, color, scale):

    (x, y) = xxx_todo_changeme
    if (color[0]+color[1]+color[2]==255*3):
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness = 4, lineType=10)
    else:
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness = 4, lineType=10)
    #cv2.line    
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), lineType=11)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    # parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=432x368, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    print("mode 0: Normal Mode \nmode 1: Debug Mode")
    mode = int(input("Enter a mode : "))

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))

    logger.debug('cam read+')

    cam = cv2.VideoCapture(1)
    # cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
    # cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    count = 0
    i = 0
    frm = 0
    y1 = [0,0]
    global height,width
    orange_color = (0,140,255)

    # Pin Setup:
    # GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    # GPIO.setup(led_pin, GPIO.OUT)  # LED pin set as output
    # GPIO.setup(but_pin, GPIO.IN)  # button pin set as input

    # Initial state for LEDs:
    # GPIO.output(led_pin, GPIO.LOW)

    while True:
        ret_val, image = cam.read()
        i =1
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        pose = humans
        if mode == 1: # Debug Mode
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        height,width = image.shape[0],image.shape[1]

        debug_info = ''

        if len(pose) > 0:

            # angle calcucations
            angle_l1 = angle_calc(find_point(pose, 6), find_point(pose, 5), find_point(pose, 1))
            angle_l2 = angle_calc(find_point(pose, 7), find_point(pose, 6), find_point(pose, 5))
            angle_r1 = angle_calc(find_point(pose, 3), find_point(pose, 2), find_point(pose, 1))
            angle_r2 = angle_calc(find_point(pose, 4), find_point(pose, 3), find_point(pose, 2))

            debug_info = str(angle_l1) + ',' + str(angle_l2) + ' : ' + str(angle_r1) + ',' + str(angle_r2)

            if muscle_pose(angle_l1, angle_l2, angle_r1, angle_r2):
                logger.debug("*** muscle Pose ***")

                # (1) create a copy of the original:
                # overlay = image.copy()
                # (2) draw shapes:
                # cv2.circle(overlay, (240, 35), 50, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
                # cv2.circle(overlay, (find_point(pose, 4)[0] - 40, find_point(pose, 4)[1] + 40), 80, (255, 241, 0), -1)
                # cv2.circle(overlay, (find_point(pose, 7)[0] - 40, find_point(pose, 7)[1] + 40), 80, (0, 241, 255), -1)
                # (3) blend with the original:
                # opacity = 0.4
                # cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
                # ser.write("light:on".encode())
                # time.sleep(1)

            elif running_man_left(angle_l1, angle_l2, angle_r1, angle_r2):
                logger.debug("*** running man left ***")
            
            elif running_man_right(angle_l1, angle_l2, angle_r1, angle_r2):
                logger.debug("*** running man right ***")
            
            elif lhand_on_relbow(angle_l1, angle_l2, angle_r1, angle_r2):
                logger.debug("*** Lhand on Relbow ***")
            
            elif fold_arms(angle_l1, angle_l2, angle_r1, angle_r2):
                logger.debug("*** fold arms ***")
                ser.write("tv:on".encode())
                time.sleep(1)

            elif hand_on_hip(angle_l1, angle_l2, angle_r1, angle_r2):
                logger.debug("*** hand on hip ***")
                ser.write("light:on".encode())
                time.sleep(1)
            
            elif hand_on_chin(angle_l1, angle_l2, angle_r1, angle_r2):
                logger.debug("*** hand on chin ***")
                ser.write("light:off".encode())
                time.sleep(1)

                # (1) create a copy of the original:
                # overlay = image.copy()
                # (2) draw shapes:
                # cv2.circle(overlay, (find_point(pose, 7)[0] + 80 ,find_point(pose, 7)[1] - 40), 80, (255, 241, 0), -1)
                # cv2.rectangle(overlay,
                            #   (find_point(pose, 7)[0] + 80,  find_point(pose, 7)[1] - 70),
                            #   (image.shape[1], find_point(pose, 7)[1] - 10),
                            #   (255, 0, 241), -1)
                # (3) blend with the original:
                # opacity = 0.4
                # cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
                # ser.write("light:off".encode())
                # time.sleep(1)
                
        image= cv2.flip(image, 1)

        if mode == 1:
            draw_str(image, (20, 50), debug_info, orange_color, 2)
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        #image =   cv2.resize(image, (720,720))

        if(frm==0):
            out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (image.shape[1],image.shape[0]))
            print("Initializing")
            frm+=1
        cv2.imshow('tf-pose-estimation result', image)
        if i != 0:
            out.write(image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

