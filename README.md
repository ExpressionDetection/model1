# Model1 - Micro Expression Nets

## Setup instructions

* Install [poetry](https://python-poetry.org/docs/)

* Run `poetry install` to install dependencies

* Test the model with a single image by running: `cd exampleScripts && python exampleUsage.py ./images/face.jpeg ../models/OuluCASIA/`
    * You can also test the model with a real-time video feed by running: `cd exampleScripts && python realtimeVideo.py ../models/CK/`

* You can serve this model by running: `python server.py`
    * This is will generate a gRPC service, more details at the `grcpPkg` folder

## CUDA setup instructions

* For this project we need `cuda==10.0` and `cuDNN==7.6.0` for Windows. For your OS other details can be found [here](https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible) and [here](https://www.tensorflow.org/install/gpu)

## How to generate or update gRCP protobuffs

* Install `grpcio-tools` globally by running `python -m pip install grpcio-tools`

* Generate gRPC code by running: `python -m grpc_tools.protoc  -I.\grcpPkg\protos --python_out=.\grcpPkg --grpc_python_out=.\grcpPkg .\grcpPkg\protos\server.proto`

# More details about this model

* Original [Github repository](https://github.com/cuguilke/microexpnet)

* The paper can be found [here](https://arxiv.org/abs/1711.07011v4)