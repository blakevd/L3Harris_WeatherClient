# Standard
import os
import sys
# Enable server function
import grpc
import argparse
# Enable Timing
import pytz
import time
import schedule
import threading
from datetime import datetime


# Create an absolute path to the common directory
common_directory = os.path.abspath(os.path.join(sys.path[0], '..', 'common'))

# Add the common directory to the system path
sys.path.insert(1, common_directory)

from google.protobuf import any_pb2
import weather_pb2
import generic_pb2
import generic_pb2_grpc

def select(server_address='73.3.127.12', server_port=50051, table_col=None, col_constraint=None):
    weather_data_array = []
    # Connect to the gRPC server
    with grpc.insecure_channel(f'{server_address}:{server_port}') as channel:
        # Create a stub (client) for the generic service
        stub = generic_pb2_grpc.DBGenericStub(channel)

        # Create a delete request

        print(table_col)
        print(col_constraint)
        
        select_request = generic_pb2.protobuf_select_request(
            keyspace="weather_data",
            table="weatherdata",
            column = table_col,
            constraint = col_constraint
        )

        # Send the delete request
        
        response = stub.Select(select_request)        
         # Loop through the protobufs field in the response
        for serialized_msg  in response.protobufs:

            # Create an instance of the WeatherData message
            weather_data = weather_pb2.WeatherData()

            # Unmarshal the binary data into the WeatherData message
            weather_data.ParseFromString(serialized_msg)

            weather_data_array.append(weather_data)
    
    return weather_data_array