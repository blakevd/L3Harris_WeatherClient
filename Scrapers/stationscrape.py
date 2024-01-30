import requests
from bs4 import BeautifulSoup

def fetch_station_codes(state_abbr):
    url = f"https://w1.weather.gov/xml/current_obs/seek.php?state={state_abbr}&Find=Find"
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all the anchor tags with href containing "stid="
        station_links = soup.find_all('a', href=lambda x: x and 'stid=' in x)

        # Extract station codes from the href attribute and store in an array
        station_codes = [link['href'].split('=')[-1] for link in station_links]
        return station_codes
    else:
        print(f"Failed to fetch the page. Status code: {response.status_code}")
        return None