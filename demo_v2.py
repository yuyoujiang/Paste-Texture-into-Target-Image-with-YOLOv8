from ultralytics import YOLO
import cv2
import numpy as np
from typing import List
import math
import argparse
import os
import datetime


class FindingMaker:
    def __init__(self, model_path, input_path, texture_path, save_dir=None):
        self.colors = [
            [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
            [255, 153, 255], [153, 204, 255], [51, 255, 51]
        ]
        self.mean_keypoints = {
            0: 'Nose', 1: 'left eye', 2: 'right eye', 3: 'left ear',
            4: 'right ear', 5: 'left shoulder', 6: 'right shoulder'
        }
        self.det_conf = 0.8
        self.texture_scale = 0.0005
        self.padding = 500

        self.model_keypoints = YOLO(model_path)
        if input_path.isnumeric():
            self.cap = cv2.VideoCapture(int(input_path))
        else:
            self.cap = cv2.VideoCapture(input_path)

        if not self.cap.isOpened():
            print('Camera opening failed！')
        self.texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
        self.texture_height, self.texture_width, _ = self.texture.shape

        self.output = None
        if save_dir is not None:
            save_dir = os.path.join(save_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.output = cv2.VideoWriter(os.path.join(save_dir, 'result.mp4'), fourcc, fps, size)

    def run(self):
        cv2.namedWindow('Maker Finding', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Maker Finding', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()
            if success:
                results = self.model_keypoints(frame)
                extract_keypoints = self.extract_keypoints(results)
                if len(extract_keypoints) == 0:
                    cv2.imshow("Maker Finding", frame)
                    if self.output is not None:
                        self.output.write(frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                # annotated_frame = self.show_point(frame, pos_left_shoulder)
                annotated_frame = self.paste_texture(frame, extract_keypoints)
                # annotated_frame = self.yolo_result_plot(annotated_frame, results[0].keypoints.data)
                cv2.imshow("Maker Finding", annotated_frame)
                if self.output is not None:
                    self.output.write(annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        self.cap.release()
        if self.output is not None:
            self.output.release()
        cv2.destroyAllWindows()

    def paste_texture(self, _frame, _pos):
        frame_h, frame_w, _ = _frame.shape

        # Create the canvas
        canvas_h, canvas_w = frame_h + self.padding, frame_w + self.padding
        canvas_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Fill the center of the canvas with the original frame
        frame_pos_w = (canvas_w - frame_w) // 2
        frame_pos_h = (canvas_h - frame_h) // 2
        canvas_image[frame_pos_h:frame_pos_h + frame_h, frame_pos_w:frame_pos_w + frame_w] = _frame

        for pos in _pos:
            # Calculate the texture size
            scale_factor = self.calculate_distance(pos['nose'], pos['left_shoulder']) * self.texture_scale
            texture_width = int(((self.texture_width * scale_factor) // 2) * 2)
            texture_height = int(((self.texture_height * scale_factor) // 2) * 2)
            texture = cv2.resize(self.texture, (texture_width, texture_height), interpolation=cv2.INTER_LINEAR)

            # Calculate the texture rotation vector
            pass

            # Insert texture according to keypoints
            h0 = int(pos['left_shoulder'][0]+(self.padding/2)-texture_height)
            w0 = int(pos['left_shoulder'][1]+self.padding/2-(texture_width/2))
            h1 = int(pos['left_shoulder'][0]+(self.padding/2))
            w1 = int(pos['left_shoulder'][1]+self.padding/2+(texture_width/2))

            insert_area = canvas_image[h0:h1, w0:w1]
            for h in range(insert_area.shape[0]):
                for w in range(insert_area.shape[1]):
                    if int(texture[h, w, 3]) != 0:
                        insert_area[h, w, :] = texture[h, w, :3]
            canvas_image[h0:h1, w0:w1] = insert_area

        # Crop the canvas and output it
        cropped_image = canvas_image[frame_pos_h:frame_pos_h + frame_h, frame_pos_w:frame_pos_w + frame_w]
        return cropped_image

    def yolo_result_plot(self, _image, _keypoints, _radius=5, _mask=None):
        if _mask is None:
            _mask = [0, 5]
        for keypoint in _keypoints:
            for i, k in enumerate(keypoint):
                if i in _mask:
                    color_k = self.colors[i]
                    x_coord, y_coord = k[0], k[1]
                    if len(k) == 3:
                        conf = k[2]
                        if conf < self.det_conf:
                            continue
                        _image = cv2.circle(
                            _image, (int(x_coord), int(y_coord)),
                            _radius, color_k, -1, lineType=cv2.LINE_AA)
        return _image

    def extract_keypoints(self, pose_result):
        keypoints = []
        if len(pose_result[0].keypoints.data.shape) == 3:
            if pose_result[0].keypoints.data.shape[1] == 0:
                return keypoints
            for data in pose_result[0].keypoints.data:
                nose = data[0]
                left_shoulder = data[5]
                if (len(left_shoulder) == 3 and left_shoulder[2] > self.det_conf and
                        len(nose) == 3 and nose[2] > self.det_conf):
                    keypoints.append(
                        {
                            'nose': [float(nose[1]), float(nose[0])],
                            'left_shoulder': [float(left_shoulder[1]), float(left_shoulder[0])]
                        }
                    )
        return keypoints

    @staticmethod
    def calculate_distance(pos1, pos2):
        distance = math.sqrt((pos2[1] - pos1[1]) ** 2 + (pos2[0] - pos1[0]) ** 2)
        return distance

    @staticmethod
    def show_point(_image, _points: List):
        for point in _points:
            _image = cv2.circle(
                _image, (int(point[0]), int(point[1])), 5,
                [0, 0, 255], -1, lineType=cv2.LINE_AA)
        return _image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./sources/yolov8m-pose.pt', type=str, help='path to model weight')
    parser.add_argument('--input', default='./sources/input_video.mov', type=str, help='usb camera id')
    parser.add_argument('--texture_path', default='./sources/makey02.png', type=str, help='path to model texture')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _args = parse_args()
    finding_maker = FindingMaker(
        _args.model_path,
        _args.input,
        _args.texture_path,
        save_dir='./save_dir'
    )
    finding_maker.run()
