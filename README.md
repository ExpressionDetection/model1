# Model1 - Micro Expression Nets

## Docker setup instructions

* Follow the steps inside the [compose](https://github.com/ExpressionDetection/compose) repository

## Manual setup instructions

* Install [poetry](https://python-poetry.org/docs/)

* Run `poetry install` to install dependencies

* Test the model with a single image by running: `cd exampleScripts && python3 exampleUsage.py ./images/face.jpg ../models/OuluCASIA/`
    * You can also test the model with a real-time video feed by running: `cd exampleScripts && python3 realtimeVideo.py ../models/CK/`

* You can serve this model by running: `python server.py`
    * This is will generate a gRPC service, more details at the `grcpPkg` folder

## CUDA setup instructions

* For this project we need `cuda==10.0` and `cuDNN==7.6.0` for Windows. For your OS other details can be found [here](https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible) and [here](https://www.tensorflow.org/install/gpu)

## How to generate or update gRCP protobuffs

* Check the documentation at [grcpPkg](https://github.com/ExpressionDetection/grcpPkg)

## More details about this model

* Original [Github repository](https://github.com/cuguilke/microexpnet)

* The paper can be found [here](https://arxiv.org/abs/1711.07011v4)