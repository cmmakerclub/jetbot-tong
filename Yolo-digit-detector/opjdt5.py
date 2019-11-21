import ipywidgets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg
from jetcam.csi_camera import CSICamera

import argparse
import json
import cv2
import numpy as np

from yolo.frontend import create_yolo
from yolo.backend.utils.box import draw_scaled_boxes
from yolo.backend.utils.annotation import parse_annotation
from yolo.backend.utils.eval.fscore import count_true_positives, calc_score

from pascal_voc_writer import Writer
from shutil import copyfile
import os
import yolo
import time

DEFAULT_CONFIG_FILE = os.path.join(yolo.PROJECT_ROOT, "svhn", "config.json")
DEFAULT_WEIGHT_FILE = os.path.join(yolo.PROJECT_ROOT, "svhn", "weights.h5")
DEFAULT_THRESHOLD = 0.3

with open('turn4/config.json') as config_buffer:
    config = json.loads(config_buffer.read())

# 2. create yolo instance & predict
yolo = create_yolo(config['model']['architecture'],
                   config['model']['labels'],
                   config['model']['input_size'],
                   config['model']['anchors'])
yolo.load_weights('model.h5')

class opimg(threading.Thread):

    # Thread class with a _stop() method.
    # The thread itself has to check
    # regularly for the stopped() condition.

    def __init__(self, *args, **kwargs):
        super(CaptureImage, self).__init__(*args, **kwargs)
        self._stop = threading.Event()
        self._camera = CSICamera(width=224, height=224)
        self._camera.running = True
        print("camera was init :)")

        # function using _stop function

    def stop(self):
        self._camera.running = False
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        while True:
            if self.stopped():
                return
            file = "dataset/{0}-{1}.jpg".format("A", time.time())
            print(file)
            cv2.imwrite(file, self._camera.value)
            time.sleep(.2)

    # def capture(self, cls, len):
    #
    #     # cv2.imwrite("dataset/{0}-{1}.jpg".format("A", time.time()), camera.value)
    #     # cap = cv2.VideoCapture(0)
    #     # print("myfunc started")
    #     # for i in range(0, len):
    #     #     ret, frame = cap.read()
    #     #     cv2.imwrite("dataset/{0}/{1}.jpg".format(cls, time.time()), frame)
    #     #     print("saved")