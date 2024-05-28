#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:25:14 2024

@author: taiwei
"""
import pandas as pd
import ast
review_data=pd.read_csv('full_airline_review.csv')
#token_data['content']=token_data['content'].apply(ast.literal_eval)
texts=list(review_data['content'])


import nltk
import re
from nltk.corpus import words, stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
print("downloading nltk packages......")
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
print("done")

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN                                                   # By default, assume noun


english_words = set(words.words())

stemmer = PorterStemmer()
def preprocessing_text(text):
    lemmatizer = WordNetLemmatizer()
    cleaned_text = re.sub(r'<.*?>', '', text)                                 # Remove HTML
    cleaned_text = re.sub(r'\d+', '', cleaned_text)                           # Remove numbers
    words = nltk. word_tokenize(cleaned_text)
    no_nonEnglish = [word.lower() for word in words if word.lower() in english_words]
    no_stopwords = [word for word in no_nonEnglish if word not in stopwords.words('english')]
    tagged = nltk.pos_tag(no_stopwords)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return ' '.join(lemmatized_words)

review_data=review_data[['header','content','rating','recommended']]
print("preprocessing tokens......")
review_data['content']=review_data['content'].apply(preprocessing_text)
texts=list(review_data['content'])
print("done")

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data):
    for text in data:
        yield tokenizer(text)
vocab = build_vocab_from_iterator(yield_tokens(texts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
review_data['token']=review_data['content'].apply(text_pipeline)

review_data.to_csv("airline_review_tokenized.csv")


# import ast
# print("loading tokenized reviews....")
# token_data=pd.read_csv('airline_review_tokenized.csv')
# token_data['token']=token_data['token'].apply(ast.literal_eval)
# print("done")
token_data=review_data

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float)
    return sequences_padded, labels, lengths

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence_tensor = torch.tensor(self.data[idx], dtype=torch.long)  # Ensure the data type is appropriate (e.g., torch.long for token indices)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)  # Adjust dtype based on what's needed for your model/loss function
        return sequence_tensor, label_tensor

import numpy as np
print("building embedding dictionary....")
embeddings_dict = {}
with open('glove.840B.300d.txt', 'r', encoding='utf8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        try:
            vector = np.asarray([i if i!='.' else '0' for i in values[1:]], dtype='float32')
        except:
            continue;
        embeddings_dict[word] = vector
print("done")


import torch.nn as nn
import torch.nn.functional as F
embedding_dim=300
vocab_size=len(vocab)+1
embedding_matrix = np.zeros((vocab_size, embedding_dim))
miss_word=[]


# Creating the embedding matrix
print("building pretrained word embedding....")
stddev=np.sqrt(2./embedding_dim)
for word, i in vocab.get_stoi().items():
    if word in embeddings_dict:
        embedding_matrix[i] = embeddings_dict.get(word)
    else:
        miss_word.append(word)
        #embedding_matrix[i] = np.random.normal(0,stddev,size=(1,embedding_dim))   # something called he normalization
print("done")
print("miss word: ", miss_word)


class TextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)  # Set pre-trained weights
        self.embedding.requires_grad = False
        self.lstm1 = nn.LSTM(embedding_dim, 50, batch_first=True)
        self.lstm2 = nn.LSTM(50, 10, batch_first=True)
        self.dropout1=nn.Dropout(0.2)
        self.fc1 = nn.Linear(10, 10)  
        self.dropout=nn.Dropout(0.2)
        self.fc2 = nn.Linear(10, 1)   

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm1_output, _ = self.lstm1(x)
        lstm2_output, _ = self.lstm2(lstm1_output)

        x, _ = nn.utils.rnn.pad_packed_sequence(lstm2_output, batch_first=True)

        idx = (lengths - 1).view(-1, 1).expand(len(lengths), x.size(2))
        idx = idx.unsqueeze(1).to(x.device)
        lstm_output = x.gather(1, idx).squeeze(1)
        x=self.dropout1(lstm_output)
        x = self.fc1(x)
        x = self.dropout(x)
        x = torch.relu(x)  
        x = self.fc2(x)


        return x


import numpy as np
print("establishing train_test data.....")
data=list(token_data['token'])
X=data
Y=np.array((token_data['recommended']=='yes')*1.)


from sklearn.model_selection import train_test_split
train_ft,test_ft,train_lbl,test_lbl = train_test_split(X,Y,test_size=0.2,random_state=114)
train_ft,valid_ft,train_lbl,valid_lbl=train_test_split(train_ft,train_lbl,test_size=0.1,random_state=514)
train_dataset = MyDataset(train_ft,train_lbl)
test_dataset = MyDataset(test_ft,test_lbl)
valid_dataset = MyDataset(valid_ft,valid_lbl)
train_loader = DataLoader(train_dataset, batch_size=512, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset,batch_size=32,  collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset,batch_size=64,collate_fn=collate_fn)
print("done")

device="cuda" if torch.cuda.is_available() else "cpu"

if 'model' in locals():
    del model
model = TextModel(vocab_size, embedding_dim, torch.tensor(embedding_matrix, dtype=torch.float)).to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam( [param for param in model.parameters() if param.requires_grad])

num_epochs = 20
print("training...")

train_loss=[]
valid_loss=[]
acc=0
for epoch in range(num_epochs):
    model.train()
    correct=0
    total=0
    for inputs, labels, lengths in train_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        outputs = model(inputs, lengths.cpu())
        
        optimizer.zero_grad()
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        model.eval()
        with torch.no_grad():
            total_loss=0
            total=0
            correct=0
            for inputs, labels, lengths in valid_loader:
                inputs,labels=inputs.to(device),labels.to(device)
                outputs=model(inputs, lengths.cpu())
                loss=criterion(outputs.squeeze(), labels)
                total_loss+=loss.item()*len(labels)
                predicts=(outputs>0.5).float().squeeze()
                correct+=(predicts==labels).float().sum()
                total+=len(labels)
            valid_loss.append(total_loss/total)
            acc=correct/total
        model.train()
    print(f'Epoch {epoch+1},  acc: {acc}')
    
print("done")
correct = 0
total = 0
print("evaluating.....")
with torch.no_grad():
    for inputs, labels, lengths in test_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        outputs = model(inputs, lengths.cpu())
        predicts=(outputs>0.5).float().squeeze()
        correct+=(predicts==labels).float().sum()
        total+=len(labels)
print(f'Accuracy of the network on the testing datapoints: {100 * correct / total} %')
print("done")


import matplotlib.pyplot as plt
plt.plot(valid_loss, color='r', label='valid')
plt.plot(train_loss, color='b', label='train')
plt.legend(title='Topics')


import torch.onnx
torch.save(model.state_dict(),"airlint_review.pth") #save the model parameters to a .pth file for python to read and load
for inputs,_,_ in train_loader:
    dummy_input=inputs
    break




torch.onnx.export(model,dummy_input,"airline_review.onnx",export_params=True, # save the model to a .onnx file for c++ to read and use
                  opset_version=10, 
                  do_constant_folding=True, 
                  input_names=['input','lengths'], 
                  output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}, 
                                'lengths': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})






