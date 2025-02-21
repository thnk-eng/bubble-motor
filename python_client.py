# bubble-motor/python_client.py

import requests

# Acquire token
token_response = requests.post(
    "http://127.0.0.1:8000/token",
    data={"username": "admin", "password": "password"}
)

if token_response.status_code != 200:
    print(f"Failed to obtain token: {token_response.text}")
    exit(1)

token = token_response.json().get("access_token")
if not token:
    print("Failed to acquire access token.")
    exit(1)

headers = {"Authorization": f"Bearer {token}"}

# Make a prediction request (REST API)
rest_response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={"input": {"input": "Hello"}},  # Adjust based on `predict` expectations
    headers=headers
)
print(f"REST API Status: {rest_response.status_code}\nResponse:\n {rest_response.json()}")

# Make a prediction request (GraphQL)
graphql_query = """
mutation Predict($input: String!) {
    predict(input_data: $input) {
        request_id
        status
        result
    }
}
"""

graphql_variables = {"input": "Hello via GraphQL"}

graphql_response = requests.post(
    "http://127.0.0.1:8000/graphql",
    json={"query": graphql_query, "variables": graphql_variables},
    headers=headers
)
print(f"GraphQL Status: {graphql_response.status_code}\nGraphQL Response:\n {graphql_response.json()}")
