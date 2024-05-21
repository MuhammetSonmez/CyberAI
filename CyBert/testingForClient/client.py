import requests
import json
import getpass
from icecream import ic
ic.configureOutput(prefix='ic|', outputFunction=lambda x: open('log.txt', 'a').write(f"{x}\n"))


with open(fr"C:\Users\{getpass.getuser()}\Desktop\Cyber Security\CyBert\datasets\classificationDataset.json" ,"r") as f:
    data = json.load(f)
    f.close()

patterns = [index['patterns'][0] for index in data['intents']]
tags = [index['tag'] for index in data['intents']]


for i in range(len(patterns)):

    url = 'http://127.0.0.1:5000/api'

    payload = {
        'input': patterns[i]
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        response_data = response.json()['answer']
        ic("query:"+patterns[i],"|response:"+ response_data , "|predicted tag:"+ tags[i])



