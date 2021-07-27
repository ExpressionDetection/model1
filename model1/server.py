from concurrent import futures
import logging
import os

import grpc

from grcppkg import server_pb2
from grcppkg import server_pb2_grpc

RPC_PORT = os.getenv("RPC_PORT", "50051")

class ModelServicer(server_pb2_grpc.ModelServicer):
    def Inference(self, request, context):
        return server_pb2.InferenceReply(prediction='Prediction, %s!' % request.image)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_pb2_grpc.add_ModelServicer_to_server(ModelServicer(), server)
    server.add_insecure_port('[::]:'+RPC_PORT)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    logging.info('Running server at port ' + RPC_PORT + '!')
    print('Running server at port ' + RPC_PORT + '!')
    serve()