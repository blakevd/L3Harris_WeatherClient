import requests

def scrape_data():
    # Send a GET request to the URL
    #url = 'https://w1.weather.gov/xml/current_obs/seek.php?state=or&Find=Find'
    url = 'https://w1.weather.gov/xml/current_obs/KSLC.xml'
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Replace &nbsp; entities with a regular space
        xml_content = response.content.decode('utf-8').replace('&nbsp;', ' ')

        # Print the XML content (for debugging purposes)
        with open('xmloutput.xml', 'w', encoding='utf-8') as file:
            file.write(xml_content)
        
    else:
        print(f"Error: Unable to fetch the data (Status Code: {response.status_code})")

# Run the scraper
scrape_data()