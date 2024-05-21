import nltk
from nltk.stem.porter import PorterStemmer
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import platform


class Internal_dataset:
  def __init__(self) -> None:
    path = ""
    if platform.system() == "Windows":
      path = r"datasets\classificationDataset.json"
    else:
      raise("platform error!")

    with open(path, "r") as f:
      data = f.read()
      f.close()

    self.data = data
  def get(self) -> dict:
    return json.loads(self.data)

class Nltk_utils:
  def __init__(self, download_punkt:bool=False)->None:
    if download_punkt:
      nltk.download('punkt')
    self.stemmer = PorterStemmer()

  def tokenize(self, sentence:str) -> list:
    return nltk.word_tokenize(sentence)

  def stem(self, word:str) -> str:
    return self.stemmer.stem(word.lower())

  def bag_of_words(self, tokenized_sentence:list, all_words:list) -> np.float32:
    """
    sentence = ['hello', 'how'  , 'are', 'you']
    words =    ['hi'   , 'hello', 'I'  , 'you', 'bye', 'thank', 'cool']
    bag =      [ 0     ,  1     ,  0   ,  1   ,  0   ,  0     ,  0]
    """

    tokenized_sentence = [self.stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, word in enumerate(all_words):
      if word in tokenized_sentence:
        bag[idx] = 1.0

    return bag


class Nlp_preprocessing(object):
  def __init__(self) -> None:
    self.intents = Internal_dataset().get()
    self.all_words = []
    self.tags = []
    self.xy = []
    self.ignore_words = ['?','!', '.', ',']
    self.nltk_utils = Nltk_utils()

    self.X_train = []
    self.y_train = []

  def preprocess(self) -> None:
    for intent in self.intents['intents']:
      tag = intent['tag']
      self.tags.append(tag)

      for pattern in intent['patterns']:
        w = self.nltk_utils.tokenize(pattern.lower())
        self.all_words.extend(w)
        self.xy.append((w,tag))

    self.all_words = [self.nltk_utils.stem(word) for word in self.all_words if word not in self.ignore_words]

    self.all_words = sorted(set(self.all_words))
    self.tags = sorted(set(self.tags))

    #print(self.all_words)
    #print(self.tags)

    for (pattern_sentence, tag) in self.xy:
      bag = self.nltk_utils.bag_of_words(pattern_sentence, self.all_words)
      self.X_train.append(bag)
      label = self.tags.index(tag)

      self.y_train.append(label)

    self.X_train = np.array(self.X_train)
    self.y_train = np.array(self.y_train)


class ChatDataset(Dataset):
  def __init__(self, n_samples, x_data, y_data) -> None:
    self.n_samples = n_samples
    self.x_data = x_data
    self.y_data = y_data

  def __getitem__(self, idx:int) -> list:
    return self.x_data[idx], torch.tensor(self.y_data[idx], dtype=torch.long)  # Cast labels to torch.long

  def __len__(self) -> int:
    return self.n_samples

# creating model


class Cyber(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes) -> None:
    super(Cyber, self).__init__()
    self.li1 = nn.Linear(input_size, hidden_size)
    self.li2 = nn.Linear(hidden_size, hidden_size)
    self.li3 = nn.Linear(hidden_size, num_classes)

    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.li1(x)
    out = self.relu(out)
    out = self.li2(out)
    out = self.relu(out)
    out = self.li3(out)

    return out


nlpp = Nlp_preprocessing()
nlpp.preprocess() # first we need preprocessing
