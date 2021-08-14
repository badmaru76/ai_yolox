# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import yolox
import os
import time
from loguru import logger
import torch
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess, vis
from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_predict(image_name, predictor, save_result):
    save_path = r"c:/"
    current_time = time.localtime()


    outputs, img_info = predictor.inference(image_name)
    print(outputs)
    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    if save_result:
        save_folder = os.path.join(
            save_path, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        save_file_name = os.path.join(save_folder, os.path.basename(image_name))
        logger.info("Saving detection result in {}".format(save_file_name))
        cv2.imwrite(save_file_name, result_image)
    return result_image


if __name__ == '__main__':
    device = "cpu"
    exp_file = 'c:\yolox_voc_s.py'
    ckpt_file = 'c:\latest_ckpt.pth.tar'

    exp = get_exp(exp_file, '')
    # confidence
    exp.test_conf = 0.25
    # nms
    exp.nmsthre = 0.45
    # image size
    exp.test_size = (640, 640)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    if device == "gpu":
        model.cuda()
    model.eval()

    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location=device)
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    trt_file = None
    decoder = None
    MY_CLASSES = ("barra", "tondo")
    predictor = Predictor(model, exp, MY_CLASSES, trt_file, decoder, device)
    print("Wait Open WebCam 1(esterna)")
    video_frame = True
    if video_frame == True:
        video = cv2.VideoCapture(1)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print("Webcam Ready")
    else:
        frame = cv2.imread("c:\prova.jpg")
    print("Start looping")
    while True:
        if video_frame == True:
            ret, frame = video.read()
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        print("AI prediction %s" %time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        ai_frame = image_predict(frame, predictor, False)
        print("AI done work %s" %time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        #key input control
        cv2.imshow('AI cam', ai_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
