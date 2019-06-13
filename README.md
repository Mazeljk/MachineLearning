# MachineLearning
Various small projects on machine learning algorithms

## DoombyDeepQ
Realize AI-player in Doom's basic scene(only three actions:left, right, shot) with deep Q-learning

## FaceRating
This programm shows how to find frontal faces in an image and judge their attractiveness. The face detector I use is **hape_predictor_68_face_landmarks.dat**.


if you are in the FaceRating folder then you can execute this program by running:

```
./face_landmark_detection.py ../test_img/test.jpg
```

## FaceRating2
Use **ResNet18** to judge a person's face attractiveness.

database from: [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)(A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction)

if you are in the FaceRating2 folder then you can execute this program by running:

```
python main.py --imagepath ./test_img/test1.jpg
```

## Genetic algorithm

Several examples using genetic algorithms


## Lane Detection
Lane detection using traditional image processing techniques

## Object Detection
Use **[SSD_mobileNet_v1_coco](https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/detection_model_zoo.md)** to detect the object

ref: modify from [tensorflow detection api demo(v1.13.0)](https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/object_detection_tutorial.ipynb)
