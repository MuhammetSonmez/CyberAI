import torch
from AI.ArtificialNerualNetworks.CyberAI import Nltk_utils, nlpp, Cyber
from AI.MlModels.MLDetective import code_detective
import random

hidden_size = 8
output_size = len(nlpp.tags)
input_size = len(nlpp.X_train[0])
model = Cyber(input_size, hidden_size, output_size)

checkpoint = torch.load("models\cyberAI.pth")
model.load_state_dict(checkpoint["model_state"])


def answer(sentence:str, bot_name:str = "CyberAI") -> list:
    print(sentence)

    sentence = sentence.lower()
    n = Nltk_utils()
    sentence = n.tokenize(sentence)
    X = n.bag_of_words(sentence, nlpp.all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = nlpp.tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in nlpp.intents['intents']:
            if tag == intent['tag']:
                code, code_description, descriptions = code_detective(sentence, tag)
                #subject = tag.replace(" ", "+")
                #print(f'{bot_name}: {random.choice(intent["responses"])}')
                #print(intent['responses'])
                return [f'{random.choice(intent["responses"])}', code, code_description, descriptions, tag] #, subject]
    else:
        #print(f"{bot_name}: I couldn't find a match in my database on this topic. If you'd like, you can contribute to my GitHub(@MuhammetSonmez) repository by collecting brief information on the term you're asking about!")
        return [f"{bot_name}: I couldn't find a match in my database on this topic. If you'd like, you can contribute to my GitHub(@MuhammetSonmez) repository by collecting brief information on the term you're asking about!", None, None, None, None]



def chat() -> None:

    print("chat started.")
    while True:
        sentence = input("say something: ")
        if sentence == "quit":
            break

        answer(sentence)



if __name__ == "__main__":
    chat()
