from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_dataset() -> dict:
    with open(r"datasets\codingDataset.json", "r") as f:
        return json.load(f)

def code_detective(user_input: str, tag: str) -> str:
    dataset = load_dataset()
    descriptions = []
    codes = []

    for topic in dataset:
        if topic['topic'] == tag:
            for sample in topic['code_samples']:
                descriptions.append(sample['description'].lower())
                codes.append(sample['code'])

    if not descriptions:
        return [None, None, None]
    else:
        vectorizer = CountVectorizer()
        vectorizer.fit(descriptions)
        desc_matrix = vectorizer.transform(descriptions)
        
        if not isinstance(user_input, str):
            user_input = str(user_input)
        
        incoming_vec = vectorizer.transform([user_input])

        cosine_similarities = cosine_similarity(incoming_vec, desc_matrix).flatten()

        most_similar_index = cosine_similarities.argmax()

        string_descriptions = ""
        for description in descriptions:
            string_descriptions += description + "\n"

        return [codes[most_similar_index], descriptions[most_similar_index], string_descriptions]


