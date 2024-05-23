# CyberAI

CyberAI is a language model trained on cybersecurity topics. This project is developed to assist users in asking and answering questions related to cybersecurity. Additionally, it provides more comprehensive answers by searching online for users' questions.

## Features

- **AI Responder:** CyberAI works by using two different models trained on cybersecurity topics:
  
  1. Artificial Neural Networks Model: Classifies incoming inputs to determine which topic they are related to.
  
  2. Cosine Similarity Model: Determines the correct response based on the topic title and the input provided.

- **Pretrained BERT Model:** Utilizes a pretrained BERT model to better understand and answer questions related to cybersecurity.(seraching on the web)

- **API Support:** A Flask-based API allows easy integration of CyberAI for users.

- **Fullstack:** Utilizes Python for both the AI and backend parts, especially libraries like PyTorch and scikit-learn. The frontend is developed using HTML, CSS, and JavaScript to provide a web interface.

## Usage

1. Create a virtual environment in the project folder and install the required dependencies:

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Start the Flask application:

```bash
python app.py
```
![Video](https://github.com/MuhammetSonmez/CyberAI/raw/main/video/cybert-beta-2-0-1.mp4)
