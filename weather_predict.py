# Standard
import sys
# Enable Server Function
import grpc
import argparse
# Enable User Predictions
import calendar
from datetime import datetime
# Enable Machine Learning
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Change directory to Routes so we can import the protobufs
current_directory = sys.path[0]
routes_directory = current_directory + '/common'
sys.path.insert(1, routes_directory)

from google.protobuf import any_pb2
import weather_pb2
import generic_pb2
import generic_pb2_grpc


# Define global variable for the model
weather_model = None
encoder_state = OneHotEncoder()
encoder_station = OneHotEncoder()
state_stations = {}

# Removes client spam from server does not print empty responses
def handle_response(response):
    if response != '':
        print(f"Server Response: {response}")

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
        handle_response(response.response)
        
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
# Returns np.arrays for training
# last_update, temp_f, relative_humidity, windchill_f
# Does not add datapoints if any are "N/A" incomplete
def get_values_from_station(server_address='localhost', server_port=50051, table_col=None, col_constraint=None):
    # Connect to the gRPC server
    with grpc.insecure_channel(f'{server_address}:{server_port}') as channel:
        # Create a stub (client) for the generic service
        stub = generic_pb2_grpc.DBGenericStub(channel)

        # Create a select request
        select_request = generic_pb2.protobuf_select_request(
            keyspace="weather_data",
            table="weatherdata",
            column=table_col,
            constraint=col_constraint
        )

        # Send the select request
        response = stub.Select(select_request)
        handle_response(response.response)
        
        # Initialize lists for each column
        last_update_values = []
        temp_f_values = []
        relative_humidity_values = []
        windchill_f_values = []

        # Loop through the protobufs field in the response
        for serialized_msg in response.protobufs:
            # Create an instance of the WeatherData message
            weather_data = weather_pb2.WeatherData()

            # Unmarshal the binary data into the WeatherData message
            weather_data.ParseFromString(serialized_msg)

            # Check if any value is "N/A"; if so, skip this data point
            if (weather_data.temp_f == "N/A" or
                weather_data.relative_humidity == "N/A" or
                weather_data.windchill_f == "N/A"):
                continue

            # Append values to respective lists
            last_update_values.append(weather_data.last_update) # Still has ' MST'
            temp_f_values.append(float(weather_data.temp_f))
            relative_humidity_values.append(float(weather_data.relative_humidity))
            windchill_f_values.append(float(weather_data.windchill_f))

        # Convert lists to NumPy arrays
        last_update_array = np.array(last_update_values) 
        temp_f_array = np.array(temp_f_values)
        relative_humidity_array = np.array(relative_humidity_values)
        windchill_f_array = np.array(windchill_f_values)

        # Return tuple of NumPy arrays
        return last_update_array, temp_f_array, relative_humidity_array, windchill_f_array

# Gets array of states, state_abbr
# From: wikipedia -> weather_client
def get_states():
    from weather_client import fetch_state_abbreviations
    return fetch_state_abbreviations()

# Gets array of stationIDs for given state_abbr
# From: weather.gov -> weather_client
def get_stations(state_abbr):
    from weather_client import fetch_station_codes
    return fetch_station_codes(state_abbr)

# Build the model layers
def build_model(input_shape):
    global weather_model
    weather_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),  # Input layer with 64 neurons and ReLU activation function
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 neurons and ReLU activation function
        tf.keras.layers.Dense(1)  # Output layer with 1 neuron (for regression task)
    ])
    # Minimizing the MSE during training means that the model's predictions are closer to the actual target values.
    # Lower MAE indicates better performance
    weather_model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Compile the model

# Setup a dictionary StateAbbr : Stations
# fits encoder_state/encoder_station to respective types
def fit_Encoders():
    global state_stations, encoder_state, encoder_station
    state_names, abbreviations = get_states()
    encoded_states = np.array(abbreviations).reshape(-1, 1)  # Reshape to column vector for encoder_state
    encoder_state.fit(encoded_states)  # Fit encoder_state with reshaped data

    for state_name, abbreviation in zip(state_names, abbreviations):
        stations = get_stations(abbreviation)
        state_stations[abbreviation] = stations  # Assign stations directly to the state abbreviation key
    
    all_stations = [station for stations in state_stations.values() for station in stations]
    encoded_stations = np.array(all_stations).reshape(-1, 1)  # Reshape to a single column for encoder_station
    encoder_station.fit(encoded_stations)  # Fit encoder_station with reshaped data
    
    # Update global variables
    encoder_state = encoder_state
    encoder_station = encoder_station

# Trains the model
# Temp_f is the target variable
# LastUpdate, Humidity, Windchill, State, Station variables used to train
def train_model():
    global weather_model
    global encoder_state
    global encoder_station
    global state_stations

    fit_Encoders()
    
    # Initialize variables for collecting data
    X_data, y_data = [], []
    
    for abbreviation in state_stations:
        values = state_stations[abbreviation]
        for station in values:

            # Get np arrays of values
            lastUpdate, temperatures, humidity, windchill = get_values_from_station(server_address=args.address, server_port=args.port, table_col="station_id", col_constraint=station)
            # Skip the station if lastUpdate is empty
            if len(lastUpdate) == 0:
                continue

            # Extract features from lastUpdate
            last_update_features = np.array([np.datetime64(date.replace(' MST', '')) for date in lastUpdate])
    
            # Extract month, day, year, and time from lastUpdate
            month = np.array([np.datetime64(date, 'M').astype(int) for date in last_update_features]).reshape(-1, 1)
            day = np.array([np.datetime64(date, 'D').astype(int) for date in last_update_features]).reshape(-1, 1)
            year = np.array([np.datetime64(date, 'Y').astype(int) for date in last_update_features]).reshape(-1, 1)
            time = np.array([np.datetime64(date, 'h').astype(int) for date in last_update_features]).reshape(-1, 1)
            
            # One-hot encode the state column
            state_encoded = np.array([abbreviation] * len(lastUpdate)).reshape(-1, 1)  # Use abbreviation for state
            state_encoded = encoder_state.transform(state_encoded).toarray()

            # One-hot encode the station column
            station_encoded = np.array([station] * len(lastUpdate)).reshape(-1, 1)
            station_encoded = encoder_station.transform(station_encoded).toarray()
            
            # Concatenate features into one array
            X = np.concatenate((state_encoded, station_encoded, month, day, year, time, humidity.reshape(-1, 1), windchill.reshape(-1, 1)), axis=1)
            
            # Collect data for later training
            X_data.append(X)
            y_data.append(temperatures)

    # Combine all data
    X_data = np.concatenate(X_data)
    y_data = np.concatenate(y_data)
    
    # Check for missing data
    if np.isnan(X_data).any() or np.isnan(y_data).any():
        print("Warning: Missing data detected during training in the input features or output labels.")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    # Build the deep learning model
    input_shape = (X_train.shape[1],)
    build_model(input_shape)
    
    # Train the model
    weather_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)

    # Evaluate the model
    evaluate_model(X_test, y_test)

# Uses remaining portion of train_test_split to evaluate the model
def evaluate_model(X_test, y_test):
    # Use the trained model to make predictions on the test data
    predictions = weather_model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print("Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

# Predict the temp_f of a future given user date and state
def predict():
    target = get_user_target()
    if target is not None: 
        target_datetime, state, station = target  # Renaming datetime variable
        print("-----")
        print("State:", state.upper(), "Station:", station, "At:", target_datetime)

        # Validate and preprocess input data
        input_features = preprocess_input_data(target_datetime, state, station)
        
        # Check for NaN values in input features
        if np.isnan(input_features).any():
            print("Warning: NaN values detected in the preprocessed input features.")
        
        # Make prediction using the trained weather_model
        predicted_temperature = weather_model.predict(input_features)
        print("Predicted Temperature:", predicted_temperature)
            
    else:
        print("User Aborted")

# Get current time and setup predictions 3, 6, 12 hours in the future of a given station
def predict_GUIGraph(target_station):
    global state_stations

    # Setup datetimes 3, 6, 12 from now
    curr_time = datetime.now()
    target_1 = compile_datetime(curr_time.year, curr_time.month, curr_time.day, (curr_time.hour + 1) % 24)
    target_2 = compile_datetime(curr_time.year, curr_time.month, curr_time.day, (curr_time.hour + 2) % 24)
    target_3 = compile_datetime(curr_time.year, curr_time.month, curr_time.day, (curr_time.hour + 3) % 24)
    target_state = find_state_for_station(target_station, state_stations)

    # Error check for state
    if (target_state == None):
        return 0, 0, 0

    # Validate and preprocess input data
    pre_result_1 = preprocess_input_data(target_1, target_state, target_station)
    pre_result_2 = preprocess_input_data(target_2, target_state, target_station)
    pre_result_3 = preprocess_input_data(target_3, target_state, target_station)

    # Check for NaN values in input features
    if np.isnan(pre_result_1).any():
        print("Warning: NaN values detected in the preprocessed input features.")
    if np.isnan(pre_result_2).any():
        print("Warning: NaN values detected in the preprocessed input features.")
    if np.isnan(pre_result_3).any():
        print("Warning: NaN values detected in the preprocessed input features.")
        
    # Make prediction using the trained weather_model
    result_1 = weather_model.predict(pre_result_1)
    result_2 = weather_model.predict(pre_result_1)
    result_3 = weather_model.predict(pre_result_1)
    return result_1, result_2, result_3

# Preprocesses target data to same format as training data
# Uses 0 as placeholder for unk windchill/humidity
def preprocess_input_data(datetime_str, state, station):
    global encoder_state
    global encoder_station

    # Convert datetime string to numpy datetime64
    datetime_np = np.array([np.datetime64(datetime_str)])
    
    # One-hot encode the state
    state_encoded = np.array([state])
    state_encoded = encoder_state.transform(state_encoded.reshape(-1, 1)).toarray()
    
    # One-hot encode the station
    station_encoded = np.array([station])
    station_encoded = encoder_station.transform(station_encoded.reshape(-1, 1)).toarray()
    
    # Extract month, day, year, and time from datetime
    month = np.array([np.datetime64(date, 'M').astype(int) for date in datetime_np]).reshape(-1, 1)
    day = np.array([np.datetime64(date, 'D').astype(int) for date in datetime_np]).reshape(-1, 1)
    year = np.array([np.datetime64(date, 'Y').astype(int) for date in datetime_np]).reshape(-1, 1)
    time = np.array([np.datetime64(date, 'h').astype(int) for date in datetime_np]).reshape(-1, 1)
    
    # Placeholder values for humidity and windchill
    humidity_placeholder = 0.0
    windchill_placeholder = 0.0
    
    # Create arrays of placeholder values for humidity and windchill
    num_samples = len(datetime_np)
    humidity = np.full((num_samples, 1), humidity_placeholder)
    windchill = np.full((num_samples, 1), windchill_placeholder)
    
    # Concatenate features into one array
    input_features = np.concatenate((state_encoded, station_encoded, month, day, year, time, humidity, windchill), axis=1)
    
    return input_features

# Given a station find the state it belongs to
def find_state_for_station(station, state_stations):
    for state, stations in state_stations.items():
        if station in stations:
            return state
    return None  # If station is not found in any state

# Converts user inputs to datetime for predictions
def compile_datetime(year, month, day, hour):
    return datetime(year, month, day, hour, 0) # Autofill 00 for minutes

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
            print(f"Invalid year input.")

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
            print("Invalid month input.")

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
            print(f"Invalid day input.")

    return day

# Validates/standardizes user input to specific time
# Returns 'None' if user aborts
def get_time():
    while True:
        try:
            user_input = input("Enter the time (24H format) (type 'back' to cancel): ")
            if user_input.lower() == 'back':
                return None
            hour = int(user_input)
            if hour < 0 or hour > 23:
                raise ValueError
            break
        except ValueError:
            print("Invalid time input.")

    return hour

# Validates/standardizes user input to specific state
# Trained model with state abbreviations (lowercase)
def get_state():
    global state_stations
    while True:
        user_input = input("Enter the 2 letter state abbr (type 'back' to cancel): ").strip().lower()
        if user_input == 'back':
            return None
        elif user_input not in state_stations:
            print("State not found.")
        else:
            return user_input

# Validates/standardizes user input to specific station
# Trained model with stations (uppercase)
def get_station(state):
    global state_stations
    while True:
        user_input = input("Enter the station (type 'back' to cancel): ").strip().upper()
        if user_input == 'BACK':
            return None
        elif user_input not in state_stations[state]:
            print(f"Station not found in {state.upper()}. Please enter a valid station.")
        else:
            return user_input

# Driver method. Gets input from user for prediction targets
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
    hour = get_time()
    if hour is None:
        return None
    state = get_state()
    if state is None:
        return None
    station = get_station(state)
    if station is None:
        return None

    return compile_datetime(year, month, day, hour), state, station

if __name__ == "__main__":
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description='Weather Prediction gRPC Client')
    parser.add_argument('--address', default='localhost', help='Address of the gRPC server')  # Add --address argument
    parser.add_argument('--port', type=int, default=50051, help='Port number for the gRPC server')  # Add --port argument
    args = parser.parse_args()
    print("---Predict Start---") # Print the initial message
    print("Client listening at port: {}".format(args.port))  
    print("Model is untrained. Beginning training.")
    train_time = datetime.now()
    train_model()

    # Check the entered flag and execute the corresponding task
    while True:
        flag = input("Enter a specific flag <Predict, Retrain, Time, Exit>: ").lower()
        if flag == 'predict':
            predict()
        elif flag == 'retrain':
            train_time = datetime.now()
            print("Beginning retraining.")
            train_model()
        elif flag == 'time':
            print("Last trained at: ", train_time)
        elif flag == 'query': # Debugging - Not main function
            column = input("Enter a specific column: ")
            constraint = input("Enter a constraint: ")
            select(server_address=args.address, server_port=args.port, table_col = column, col_constraint = constraint)
        elif flag == 'exit':
            print("Exited Client.")
            break
        elif flag == 'debug': # Updated to test various methods
            print(find_state_for_station("KPDX", state_stations))
            print(predict_GUIGraph("KSLC"))
            break
        else:
            print("Invalid flag. Please enter a valid flag.")