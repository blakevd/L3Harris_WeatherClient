# Standard
import os
import sys
# Enable Server Function
import grpc
import argparse
# Enable User Predictions
import datetime
import calendar
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


# Converts user inputs to datetime for predictions
def compile_datetime(year, month, day, hour, minute):
    return datetime.datetime(year, month, day, hour, minute)

# Validates/standardizes user input to specific year
# Returns 'None' if user aborts
def get_year():
    min_year = 2024  # Minimum allowed year
    max_year = 2024  # Maximum allowed year

    while True:
        try:
            user_input = input(f"Enter the year ({min_year}-{max_year}) (type 'back' to cancel): ")
            if user_input.lower() == 'back':
                return None
            year = int(user_input)
            if year < min_year or year > max_year:
                raise ValueError
            break
        except ValueError:
            print(f"Invalid year. Please enter a year between {min_year} and {max_year}.")

    return year

# Validates/standardizes user input to specific month
# Returns 'None' if user aborts
def get_month():
    while True:
        try:
            user_input = input("Enter the month (e.g., 'December', 'Dec', or 12) (type 'back' to cancel): ")
            if user_input.lower() == 'back':
                return None
            if user_input.isdigit():
                month = int(user_input)
                if month < 1 or month > 12:
                    raise ValueError
            else:
                month_abbr = user_input[:3].capitalize()
                month_full = user_input.capitalize()
                month = list(calendar.month_abbr).index(month_abbr)
                if month == 0:
                    month = list(calendar.month_name).index(month_full)
                    if month == 0:
                        raise ValueError
            break
        except ValueError:
            print("Invalid month input. Please enter a valid month")

    return month

# Validates/standardizes user input to specific day within specific month
# Returns 'None' if user aborts
def get_day(year, month):
    while True:
        try:
            user_input = input(f"Enter the day (type 'back' to cancel): ")
            if user_input.lower() == 'back':
                return None
            day = int(user_input)
            max_day = calendar.monthrange(year, month)[1]
            if day < 1 or day > max_day:
                raise ValueError
            break
        except ValueError:
            print(f"Invalid day. Please enter a day between 1 and {max_day} for the given month and year.")

    return day

# Validates/standardizes user input to specific time
# Returns 'None' if user aborts
def get_time():
    while True:
        try:
            user_input = input("Enter the time (in 24-hour format HH:MM) (type 'back' to cancel): ")
            if user_input.lower() == 'back':
                return None, None
            hour, minute = map(int, user_input.split(':'))
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                raise ValueError("Invalid time format. Please enter a valid time in 24-hour format.")
            break
        except ValueError:
            print("Invalid time input. Please enter a valid time")

    return hour, minute

# Driver method. Gets input from user for prediction target date
# Returns 'None' if user aborts
def get_user_target():
    year = get_year()
    if year is None:
        return None
    month = get_month()
    if month is None:
        return None
    day = get_day(year, month)
    if day is None:
        return None
    hour, minute = get_time()
    if hour is None or minute is None:
        return None

    return compile_datetime(year, month, day, hour, minute)


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
            target_datetime = get_user_target()
            if (target_datetime != None): # Debugging
                print(target_datetime)
            else:
                print("User Aborted")
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