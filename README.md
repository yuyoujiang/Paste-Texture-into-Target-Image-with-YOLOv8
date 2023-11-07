# Paste Texture into Target Image with YOLOv8
This is a image fusion demo with YOLOv8. 
You can use this project to paste a specific texture on the left shoulder of pedestrian objects in the input image. 
Specifically, using YOLOv8 to detect keypoint information in the input image, adjust the size and position of the texture based on the keypoints.

This has been tested and deployed on a [reComputer Jetson J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html). 
However, you can use any NVIDIA Jetson device to deploy this demo.

## Introduction

Select the position coordinates of the nose and left shoulder from the inference results of YOLOv8-Pose. 
Determine the size of the texture based on the distance between the nose and shoulder. 
Determine the position of the texture based on the location of the left shoulder. 
Based on above-mentioned mechanism, we can fusion two images and produce a very interesting application.

## Installation

- **Step 1:** Flash JetPack OS to reComputer Jetson device [(Refer to here)](https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/).

- **Step 2:** Access the terminal of Jetson device, install pip and upgrade it

```sh
sudo apt update
sudo apt install -y python3-pip
pip3 install --upgrade pip
```

- **Step 3:** Clone the following repo

```sh
git clone https://github.com/ultralytics/ultralytics.git
```

- **Step 4:** Open requirements.txt

```sh
cd ultralytics
vi requirements.txt
```

- **Step 5:** Edit the following lines. Here you need to press i first to enter editing mode. Press ESC, then type :wq to save and quit

```sh
# torch>=1.7.0
# torchvision>=0.8.1
```

**Note:** torch and torchvision are excluded for now because they will be installed later.

- **Step 6:** Install the necessary packages

```sh
pip3 install -e .
```

- **Step 7:** If there is an error in numpy version, install the required version of numpy

```sh
pip3 install numpy==1.20.3
```

- **Step 8:** Install PyTorch and Torchvision [(Refer to here)](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/#install-pytorch-and-torchvision).

- **Step 9:** Run the following command to make sure yolo is installed properly

```sh
yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg' 
```

- **Step 10:** Clone exercise counter demo

```sh
git clone https://github.com/yuyoujiang/Paste-Texture-into-Target-Image-with-YOLOv8.git
```

## Prepare Model File

YOLOv8-pose pretrained pose models are PyTorch models and you can directly use them for inferencing on the Jetson device. However, to have a better speed, you can convert the PyTorch models to TensorRT optimized models by following below instructions.

- **Step 1:** Download model weights in PyTorch format [(Refer to here)](https://docs.ultralytics.com/tasks/pose/#models).

- **Step 2:** Execute the following command to convert this PyTorch model into a TensorRT model 

```sh
# TensorRT FP32 export
yolo export model=<path to model>/yolov8m-pose.pt format=engine device=0

# TensorRT FP16 export
yolo export model=<path to model>/yolov8m-pose.pt format=engine half=True device=0
```

**Tip:** [Click here](https://docs.ultralytics.com/modes/export) to learn more about yolo export 

- **Step 3:** Organize resource files(model weights and textures).
```sh
mkdir sources

demo
├── MakerFinding.py
├── ultralytics
└── soueces
    ├── makey02.png
    ├── yolov8m-pose.pt
    ... 
```

## Let's Run It!

```sh
python3 MakerFinding.py --model_path ./sources/yolov8m-pose.pt --input 0 --texture_path ./sources/makey02.png
```

![result 00_00_00-00_00_30](https://github.com/yuyoujiang/exercise-counting-with-YOLOv8/assets/76863444/414e1cd1-ab7d-4ca6-91e4-c8a948fe55ae)

## References

[https://github.com/ultralytics/](https://github.com/ultralytics/)  
[https://wiki.seeedstudio.com](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/)
