from concurrent import futures
from time import gmtime, strftime 
import tensorflow as tf
import numpy as np
import cv2
import sys
import logging
import os
import io
import json
import PIL.Image as Image

import grpc
from grpc_reflection.v1alpha import reflection

from grcppkg import server_pb2, server_pb2_grpc

from model1.MicroExpNet import *

# Import the xml files of frontal face and eye
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detectFaces(img):
	# Convertinto grayscale since it works with grayscale images
	gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect the face
	faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

	if len(faces):
		return faces[0] 
	else:
		return [-13, -13, -13, -13]

# Detects the face and eliminates the rest and resizes the result img
def segmentFace(img, imgXdim, imgYdim):
	# Convert image to numpy array
	img = np.array(img)

	# Detect the face
	(p,q,r,s) = detectFaces(img)

	# Return the whole image if it failed to detect the face
	if p != -13: 
		img = img[q:q+s, p:p+r]

	# Crop & resize the image
	img = cv2.resize(img, (256, 256)) 	
	img = img[32:256, 32:256]
	img = cv2.resize(img, (imgXdim, imgYdim)) 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def get_time():
	return strftime("%a, %d %b %Y %X", gmtime())   


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    RPC_PORT = os.getenv("RPC_PORT", "50051")
    MODEL_ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", "/app/model1/models/OuluCASIA") 

    # Model labels
    labels = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

    imgYdim = 84
    imgXdim = 84
    nInput = imgXdim*imgYdim # Since RGB is transformed to Grayscale

    # tf Graph input
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, nInput])

    # Construct model
    classifier = MicroExpNet(x)

    # Deploy weights and biases for the model saver
    weights_biases_deployer = tf.compat.v1.train.Saver({"wc1": classifier.w["wc1"], \
                                        "wc2": classifier.w["wc2"], \
                                        "wfc": classifier.w["wfc"], \
                                        "wo": classifier.w["out"],   \
                                        "bc1": classifier.b["bc1"], \
                                        "bc2": classifier.b["bc2"], \
                                        "bfc": classifier.b["bfc"], \
                                        "bo": classifier.b["out"]})

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False)) as sess:

        sess.run(tf.compat.v1.global_variables_initializer())
        weights_biases_deployer.restore(sess, tf.train.latest_checkpoint(MODEL_ARTIFACT_DIR))

        class ModelServicer(server_pb2_grpc.ModelServicer):
            def Inference(self, request, context):
                # Read the image
                image = Image.open(io.BytesIO(request.image))
                # image.save("/app/test.png") # For debugging
                imageArr = segmentFace(image, imgXdim, imgYdim)
                # cv2.imwrite("/app/test.png", imageArr) # For debugging
                imageArr = np.reshape(imageArr, (1, imgXdim*imgYdim))
                imageArr = imageArr.astype(np.float32)

                probabilities = sess.run([classifier.pred], feed_dict={x: imageArr})
                # print(probabilities)
                argmax = np.argmax(probabilities)
                # print("[" + get_time() + "] Emotion: " + labels[argmax])

                return server_pb2.InferenceReply(prediction=json.dumps({
                                                                        "labels": labels,
                                                                        "probabilities": probabilities[0][0] * 100, # Convert probabilities from 0 to 100
                                                                        "topProbabilityIndex": argmax
                                                                        }, cls=NumpyEncoder))

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        server_pb2_grpc.add_ModelServicer_to_server(ModelServicer(), server)
        # the reflection service will be aware of "ModelServicer" and "ServerReflection" services.
        SERVICE_NAMES = (
            server_pb2.DESCRIPTOR.services_by_name['Model'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port('[::]:'+RPC_PORT)
        server.start()
        server.wait_for_termination()

        logging.info('Running server at port: ' + RPC_PORT) 