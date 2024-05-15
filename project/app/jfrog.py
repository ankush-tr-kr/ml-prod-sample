import requests

url = "https://krogertechnology.jfrog.io/artifactory/api/repositories"
headers = {
    "X-JFrog-Art-Api": "AKCpBseMqpibWHcMmt6UfmrYuwdmW3U2DMPwVsXSwQmVykyWim8sk1J24wZvWPq8UQfbbvwwv"
}

response = requests.get(url, headers=headers)
print(response.text)