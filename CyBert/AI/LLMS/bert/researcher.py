import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
import re
import spacy
import transformers
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings

transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=FutureWarning)

# Web Scraping
def web_scraping(url:str, subject:str) -> str:
    url = url+subject.lower()
    response = requests.get(url)
    if response.status_code != 200:
        print("check your network...")
        exit()
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])

    if f"The page" in text and  "does not exist" in text:
        soup = BeautifulSoup(response.text, 'html.parser')
        div = soup.find('div', class_='mw-search-result-heading')
        a_tag = div.find('a')
        url = a_tag['href']
        
        response = requests.get("https://en.wikipedia.org" + url)
        if response.status_code != 200:
            print("check your network...")
            exit()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
            
    return text



# Text pre-processing
def preprocess_text(text:str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Natural Language Processing
def nlp_processing(text:str) -> list:
    # Tokenization ve lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Information Extraction
def extract_information(text:str):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return doc

# Question Answering
def answer_question(question:str, text:str) -> str:
    model = transformers.BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    max_seq_length = 512
    input_ids = tokenizer.encode(question, text, add_special_tokens=True, return_tensors="pt")
    if input_ids.shape[1] > max_seq_length:
        input_ids = input_ids[:, :max_seq_length]

    outputs = model(input_ids)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    start_scores = start_scores.detach().numpy()
    end_scores = end_scores.detach().numpy()
    answer_start = np.argmax(start_scores)
    answer_end = np.argmax(end_scores) + 1

    max_answer_length = 250
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end])
    if len(answer_tokens) > max_answer_length:
        answer_tokens = answer_tokens[:max_answer_length]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer




def convert_rdf_to_nx_graph(graph):
    nx_graph = nx.DiGraph()
    for subject, predicate, obj in graph:
        nx_graph.add_node(subject)
        nx_graph.add_node(obj)
        predicate_label = str(predicate).split('/')[-1]
        nx_graph.add_edge(subject, obj, label=predicate_label)
    return nx_graph

def draw_graph(nx_graph):
    
    pos = nx.spring_layout(nx_graph, k=0.15, iterations=20)
    labels = nx.get_edge_attributes(nx_graph, 'label')

    plt.figure(figsize=(100, 100))
    nx.draw(nx_graph, pos, with_labels=True, node_size=100, node_color="skyblue", font_size=12, arrows=True)
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=labels, font_size=10)
    plt.show()


def search(subject:str, question:str) -> str:
    url = "https://en.wikipedia.org/w/index.php?search="
    text = web_scraping(url, subject)
    
    processed_text = preprocess_text(text)

    answer = answer_question(question, processed_text)

    return answer




if __name__ == "__main__":
    subject = "Cross+Site+Scripting+(XSS)"
    question = "what is xss ?"
    result = search(subject, question)
    print(result)



