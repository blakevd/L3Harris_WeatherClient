import requests
from lxml import etree

def fetch_weather_data(station_code):
    # Construct the XML endpoint URL based on the station code
    xml_url = f'https://w1.weather.gov/xml/current_obs/{station_code}.xml'

    # Send a GET request to the specified XML endpoint
    response = requests.get(xml_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        try:
            # Parse the XML content
            root = etree.fromstring(response.content)

            # Extract information from the XML data
            # Check if values exist
            station_name_element = root.find('location')
            station_name = station_name_element.text if station_name_element is not None else "N/A"

            temperature_element = root.find('temp_f')
            temperature = float(temperature_element.text) if temperature_element is not None else "N/A"

            weather_element = root.find('weather')
            weather = weather_element.text if weather_element is not None else "N/A"

            observation_time = root.find('observation_time_rfc822')
            last_update = observation_time.text if observation_time is not None else "N/A"

            # Return data
            return station_name, temperature, weather, last_update

        except etree.XMLSyntaxError as e:
            print(f"Error parsing XML: {e}")
    else:
        print(f"Error: Unable to fetch the weather data (Status Code: {response.status_code})")
        return None, None, None, None