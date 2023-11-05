from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
# Set up the web driver using ChromeDriverManager
driver = webdriver.Chrome(ChromeDriverManager().install())
from selenium.webdriver.common.keys import Keys


# Define the list of locations
locations = [
    {'Latitude': '12.34', 'Longitude': '56.78'},
    {'Latitude': '0.123', 'Longitude': '-45.678'},
    # Add more locations as needed
]

# Define the URL of the web page
url = 'http://192.168.144.140:5000/'

# Open the web page
driver.get(url)

# Wait for the page to load
time.sleep(2)

# Find the input fields and submit button
latitude_field = driver.find_element_by_id('Latitude')
longitude_field = driver.find_element_by_id('Longitude')
submit_button = driver.find_element_by_css_selector('.btn.btn-primary')

# Iterate through the locations and automate input and submission
for location in locations:
    latitude_field.clear()
    latitude_field.send_keys(location['Latitude'])
    longitude_field.clear()
    longitude_field.send_keys(location['Longitude'])
    submit_button.click()
    
    # Wait for the results to load (adjust the wait time as needed)
    time.sleep(5)
    
    # Extract and print the results (you need to locate the results element on your page)
    results_element = driver.find_element_by_css_selector('your_results_selector_here')
    print(f"Results for location: {location['Latitude']}, {location['Longitude']} - {results_element.text}")

# Close the web browser
driver.quit()
