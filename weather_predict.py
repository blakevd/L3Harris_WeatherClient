# Standard
import os
import sys
# Enable Server Function
import grpc
import argparse
# Enable Machine Learning
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Change directory to Routes so we can import the protobufs
current_directory = sys.path[0]
routes_directory = current_directory + '/common'
sys.path.insert(1, routes_directory)

from google.protobuf import any_pb2
import weather_pb2
import generic_pb2
import generic_pb2_grpc


# Query values in the Database (For Debugging)
def select(server_address='localhost', server_port=50051, table_col=None, col_constraint=None):
    # Connect to the gRPC server
    with grpc.insecure_channel(f'{server_address}:{server_port}') as channel:
        # Create a stub (client) for the generic service
        stub = generic_pb2_grpc.DBGenericStub(channel)

        # Create a select request
        select_request = generic_pb2.protobuf_select_request(
            keyspace="weather_data",
            table="weatherdata",
            column = table_col,
            constraint = col_constraint
        )

        # Send the select request
        response = stub.Select(select_request)
        print(f"Server Response: {response.response}")
        
         # Loop through the protobufs field in the response
        for serialized_msg  in response.protobufs:

            # Create an instance of the EducationData message
            weather_data = weather_pb2.WeatherData()

            # Unmarshal the binary data into the EducationData message
            weather_data.ParseFromString(serialized_msg)

            # Print specific fields from Weather_Data/weatherdata Keyspace/table
            print(f"State: {weather_data.state} : {weather_data.station_id}, "
            f"Last Update: {weather_data.last_update}, "
            f"F Temp: {weather_data.temp_f} : {weather_data.weather}")

# Query stations in the Database
# Cleans out "N/A" strings
# Returns np.array of last_update, temp_f for training
def getValuesFromStation(server_address='localhost', server_port=50051, table_col=None, col_constraint=None):
    # Connect to the gRPC server
    with grpc.insecure_channel(f'{server_address}:{server_port}') as channel:
        # Create a stub (client) for the generic service
        stub = generic_pb2_grpc.DBGenericStub(channel)

        # Create a select request
        select_request = generic_pb2.protobuf_select_request(
            keyspace="weather_data",
            table="weatherdata",
            column = table_col,
            constraint = col_constraint
        )

        # Send the select request
        response = stub.Select(select_request)
        print(f"Server Response: {response.response}")
        
        last_update_values = []
        temp_f_values = []
         # Loop through the protobufs field in the response
        for serialized_msg  in response.protobufs:

            # Create an instance of the EducationData message
            weather_data = weather_pb2.WeatherData()

            # Unmarshal the binary data into the EducationData message
            weather_data.ParseFromString(serialized_msg)

            # Check if temp_f value is "N/A", if not, append both temp_f and last_update values
            if weather_data.temp_f != "N/A":
                temp_f_values.append(float(weather_data.temp_f))
                last_update_values.append(weather_data.last_update)

        # Convert lists to NumPy arrays
        last_update_array = np.array(last_update_values)
        temp_f_array = np.array(temp_f_values)

        # Return tuple of NumPy arrays
        return last_update_array, temp_f_array

# Gets array of states, state_abbr
# From: wikipedia -> weather_client
def getStates():
    from weather_client import fetch_state_abbreviations
    return fetch_state_abbreviations()

# Gets array of stationIDs for given state_abbr
# From: weather.gov -> weather_client
def getStations(state_abbr):
    from weather_client import fetch_station_codes
    return fetch_station_codes(state_abbr)

def train():
        # Get data from weather_client
        state_names, abbreviations = getStates()
        for state, abbreviation in zip(state_names, abbreviations):
            # Get data from weather_client
            stations = getStations(abbreviation)
            for station in stations:
                # Format into np arrays
                lastUpdate, temperatures = getValuesFromStation(server_address=args.address, server_port=args.port, table_col = "station_id", col_constraint = station)
                print(temperatures)
                print(lastUpdate)

        
        #build model
        #train

        {
        #     # Store 'station_ID' in 'stations' dictionary as Key if it doesn't exist
        #     if weather_data.station_ID not in stations:
        #         stations[weather_data.station_ID] = {}  # Initialize the dictionary for this station ID if it's not already present

        #     # Store 'temp_f' in station_ID dictionary as Key
        #     # Store values of temp_f
        #     if 'temp_f' not in stations[weather_data.station_ID]:
        #         stations[weather_data.station_ID]['temp_f'] = np.array([weather_data.temp_f])  # Initialize the numpy array
        #     else:
        #         stations[weather_data.station_ID]['temp_f'] = np.append(stations[weather_data.station_ID]['temp_f'], weather_data.temp_f)

        #     # Debugging print
        # for station_id, station_data in stations.items():
        #     print(f"Station ID: {station_id}")
        #     for key, value in station_data.items():
        #         print(f"Key: {key}, Values: {value}")
        }


# Unfinished test code
def test():
    inputs = keras.Input(shape=(784,), name="digits")                       # Input layer, named Digits for debugging
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)         # Creates first dense (fully connected layer) in neural network
                                                                            # 64 Neurons, relu introduces non-linearity, named dense_1, connected to inputs
    x = layers.Dense(64, activation="relu", name="dense_2")(x)              # Creates second dense layer, connected to x
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x) # Creates output layer, 10 neurons, softmax convert raw scores to probability, connected to x

    model = keras.Model(inputs=inputs, outputs=outputs)                     # Creates whole Model object, specifies inputs/output layers (encaspulates dense_1/2)

if __name__ == "__main__":
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description='Weather Prediction gRPC Client')
    parser.add_argument('--address', default='localhost', help='Address of the gRPC server')  # Add --address argument
    parser.add_argument('--port', type=int, default=50051, help='Port number for the gRPC server')  # Add --port argument
    args = parser.parse_args()
    print("---Predict Start---")
    print("Client listening at port: {}".format(args.port))  # Print the initial message

    # Check the entered flag and execute the corresponding task
    while True:
        flag = input("Enter a specific flag <Predict, Exit>: ").lower()
        if flag == 'predict':
            pass
        elif flag == 'build': # Debugging - Not main function
            pass
        elif flag == 'train': # Debugging
            train()
        elif flag == 'query': # Debugging - Not main function
            column = input("Enter a specific column: ")
            constraint = input("Enter a constraint: ")
            select(server_address=args.address, server_port=args.port, table_col = column, col_constraint = constraint)
        elif flag == 'exit':
            print("Exited Client.")
            break
        else:
            print("Invalid flag. Please enter a valid flag.")