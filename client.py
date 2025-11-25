# API Client for Bitcoin Price Predictor

import requests
import sys

# Data to be sent to the API
payload = {"Model": "Machine Learning"}

# API endpoint URL
api_url = "http://localhost:3000/predict"

try:
    # Make the request to the API
    response = requests.post(api_url, json = payload, timeout = 10) # 10-second timeout

    # Raise an exception for bad status codes (4xx or 5xx)
    response.raise_for_status()

    # Parse the JSON response
    response_data = response.json()

    # Print the response
    print('\nAccessing the Project 2 API to Predict Bitcoin Price!')
    print('\nAPI Response:\n')
    print(response_data)

except requests.exceptions.ConnectionError:
    print(f"\n[ERROR] Could not connect to the API at {api_url}.")
    print("Please ensure the API server is running.", file = sys.stderr)

except requests.exceptions.Timeout:
    print(f"\n[ERROR] The request to {api_url} timed out.", file = sys.stderr)

except requests.exceptions.HTTPError as http_err:
    print(f"\n[ERROR] HTTP error occurred: {http_err}")
    print(f"Response content: {response.text}", file = sys.stderr)

except requests.exceptions.RequestException as err:
    print(f"\n[ERROR] An unexpected error occurred: {err}", file = sys.stderr)

finally:
    print('\nThank You For Using This API!\n')