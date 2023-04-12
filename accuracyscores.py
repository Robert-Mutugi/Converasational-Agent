import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import word2vec
from transformers import AutoModel, DistilBertTokenizer
import torch
from numpy.linalg import norm
import math
import re
from collections import Counter


df = pd.read_csv('groundtruth.txt', sep='?', header=None)
dg = pd.read_csv('predicted.txt', header=None)


xdata = df[0]
ydata = dg[0]
x = []
y = []
list(x)
#print(xdata,ydata)

for i, target in zip(xdata, ydata):
    i = i.split()
    target = target.split()
    y.append(target)
    x.append(i)
    
acc=0
index = 0 
for x1, target in zip(x,y):
    acc += sentence_bleu(x1, target)

print('Bleu Score is', acc/len(x))



label = re.compile(r"\w+")


def cosine(sen1, sen2):
    simi = set(sen1.keys()) & set(sen2.keys())
    numerator = sum([sen1[x]*sen2[x] for x in simi])

    sum1 = sum([sen1[x]**2 for x in list(sen1.keys())])
    sum2 = sum([sen2[x]**2 for x in list(sen2.keys())])

    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    return 0.0 if not denominator else float(numerator)/denominator


def text_vector(text):
    words = label.findall(text)
    return Counter(words)

avg_cosine = 0
for x,y in zip(xdata, ydata):
    #print(type(x))
    lines1 = text_vector(x)
    lines2 = text_vector(y)
    #print(lines1, lines2)
    avg_cosine += cosine(lines1, lines2)
print('Average cosine similarity is: ', avg_cosine/len(xdata))

def word_overlap(sen1, sen2):
    predicted = []
    count = 0
    for y in sen2:
        substring = y.split(" ")
        predicted.append(substring)
    #print(predicted)

    for sub, string in zip(predicted, xdata):
        #print(sub, string)
        for word in sub:
            for i in range(len(string)):
                i = string.find(word)
                if i != -1:
                    count += 1
                else:
                    break
    print('The word overlap is:' ,count/len(xdata))


output = word_overlap(xdata, ydata)
print(len(xdata), len(ydata))




def word_emb(xdata, ydata):
    cosimi = 0
    for x,y in zip(xdata, ydata):
        checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = AutoModel.from_pretrained(checkpoint)
        inputx = tokenizer(x, return_tensors="pt")
        inputy = tokenizer(y, return_tensors="pt")
        outputsx = model(**inputx)
        outputsy = model(**inputy)
        #print(outputsx.last_hidden_state.shape)
        #print(outputsx["last_hidden_state"][0,-1])
        emb1 = outputsx["last_hidden_state"][0,-1]
        emb2 = outputsy["last_hidden_state"][0, -1]
        emb1 = emb1.detach().numpy()
        emb2 = emb2.detach().numpy()
        cosim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
        #print(range(len(emb1)))
        cosimi += cosim
    print(cosimi/len(x))
        

word_emb(xdata,ydata)