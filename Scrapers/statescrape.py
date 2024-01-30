import requests
from bs4 import BeautifulSoup

def fetch_state_abbreviations():
    url = "https://en.wikipedia.org/wiki/List_of_U.S._state_abbreviations"
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        state_names = []
        abbreviations = []
        states_found = 0

        # The state abbreviations are in a table on the Wikipedia page
        table = soup.find('table', {'class': 'wikitable'})

        if table:
            # Iterate through rows of the table (skipping header row)
            for row in table.find_all('tr'):
                columns = row.find_all(['td', 'th'])

                # Check if the row has at least four columns
                if len(columns) >= 4:
                    state_name = columns[0].text.strip()

                    if state_name and len(state_name) <= 20:
                        # Exclude Washington, D.C.
                        if state_name != "District of Columbia":
                            abbreviation = columns[3].text.strip()
                            state_names.append(state_name)
                            abbreviations.append(abbreviation.lower())
                            states_found += 1

                            # Stop after 50 states (excluding D.C.)
                            if states_found == 50:
                                break
        
        return state_names, abbreviations

    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None, None
