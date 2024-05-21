import json

with open("datasets/classificationDataset.json", "r") as f:
    data = json.load(f)

removeList = "Name", "Greeting", "Goodbye", "Thanks", "Help"
tags = [index['tag'] for index in data['intents']]
for i in removeList:
    tags.remove(i)

tags = str(tags)
tags = tags.replace("'", "").replace("[","").replace("]", "")


print(tags)
