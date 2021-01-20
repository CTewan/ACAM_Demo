# Action Classification Repository

This repo is referenced from https://github.com/oulutan/ACAM_Demo/. It is used to identify and track a particular person in the video, and subsequently classify his/her actions.

# Installation

1. Clone this repo recursively

```bash
git clone --recursive https://github.com/CTewan/MOE-ICAN
```

2. Install the required libraries

```bash
pip install -r requirements.txt
```

3. Compile Tensorflow Object Detection API

```bash
protoc object_detection/models/research/object_detection/protos/*.proto --python_out=.
```

If you do not have protobuf, download it from https://github.com/protocolbuffers/protobuf/releases and unzip it. 

4. Download and extract Tensorflow Object Detection models into object_detection/weights/ from:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md

5. Download DeepSort re-id model into object_detection/deep_sort/weights/ from their author's drive:
https://drive.google.com/open?id=1m2ebLHB2JThZC8vWGDYEKGsevLssSkjo

6. Download the action detector model weights into action_detection/weights/ from following link:
https://drive.google.com/open?id=138gfVxWs_8LhHiVO03tKpmYBzIaTgD70

# How to Run

There are 2 scripts in this repo.

1. ```detect_actions.py```

Runs script on a single input video.

Arguments
* -v (--video_path): The path to the video and if it is not provided, webcam will be used.
* -d (--display): The display flag where the results will be visualized using OpenCV.

Example:
```bash
python detect_actions.py --display True --video_path input_videos/{name}.mp4
```

2. ```run_all.py```

Runs script on all the videos available in folder sequentially.

Arguments
* -f (--video_folder): The path to the folder containing all input videos in mp4 format.
* -d (--display): The display flag where the results will be visualized using OpenCV.

Example:
```bash
python run_all.py --display False --video_path input_videos
```

# References
1. https://github.com/oulutan/ACAM_Demo/