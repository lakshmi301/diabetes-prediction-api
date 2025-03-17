import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [2, 120, 70, 20, 79, 32.0, 0.4, 25]}

response = requests.post(url, json=data)
print(response.json())
