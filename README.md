# Micro Expression Nets

## Setup instructions

* Install [poetry](https://python-poetry.org/docs/)

* Run `poetry install` to install dependencies

* Test the model with a single image by running: `python exampleUsage.py ./images/face.jpeg ./models/OuluCASIA/`
    * You can also test the model with a real-time video feed by running: `python realtimeVideo.py ./models/CK/`

## CUDA setup instructions

* For this project we need `cuda==10.0` and `cuDNN==7.6.0` for Windows. For your OS other details can be found [here](https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible) and [here](https://www.tensorflow.org/install/gpu)