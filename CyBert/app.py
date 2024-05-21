from flask import Flask, request, jsonify, render_template
import Ichat
from icecream import ic
from AI.LLMS.bert.researcher import search

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/bert', methods=['POST'])
def ask_to_bert():
    data = request.get_json()
    user_input = data.get('input')
    _, _, _, _, tag = Ichat.answer(user_input)
    if tag != None:
        subject = tag.replace(" ", "+")
        result = search(subject, user_input)
        return jsonify({'bert_answer':result})
    else:
        return jsonify({'bert_answer':"I don't understand"})
    
    

@app.route('/api', methods=['POST'])
def receive_input():
    data = request.get_json()
    user_input = data.get('input')
    #answer, code, code_description, descriptios, subject = Ichat.answer(user_input)
    #result = search(subject, user_input)
    answer, code, code_description, descriptios, tag = Ichat.answer(user_input)
    ic(answer, code, code_description, descriptios)

    if code_description == None and code == None and descriptios == None:
        return jsonify({'answer': answer})

    print(type(descriptios))
    print(descriptios)
    return jsonify({'answer': answer, 'code': code_description+"\n"+code,  "descriptions":f"{tag}"+"\n"+descriptios})

if __name__ == '__main__':
    app.run(debug=True)
