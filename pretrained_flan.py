import gzip
import shutil
import torch
import tensorflow as tf
import os as os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import evaluate
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback

folder_name = r'C:/Users/rmuri/Desktop/Project work/training_data'
file_names = [os.path.join(folder_name, name) for name in os.listdir(folder_name)]

'''for path in file_names[:50]:
    with gzip.open(path) as f:
        with open('training_set.jsonl', 'wb') as g:
            shutil.copyfileobj(f, g)'''

all_files = pd.read_json(path_or_buf='training_set.jsonl',lines=True)

df = all_files['utterances']
#print(df)
text = []

def flatten(data):
    for ele in data:
        if type(ele)==list:
            flatten(ele)
        else:
            text.append(ele)

flatten(df)

doc_text = pd.Series(text)
ignore = "Hi, I'm your automated assistant."
i= 0
x = []
y = []

for j in df[:50]:
    for k in j:
        if k.find(ignore)==0: continue
        if i%2==0: x.append(k)
        if i%2!=0: y.append(k)
        i = i+1

list(x)
list(y)
#print(doc_text)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model= T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

#xtrain_encode = tokenizer(x_train, truncation=True, padding=True)
#xtest_encode = tokenizer(x_test, truncation=True, padding=True)

#Input
question = input('Hi, how can')
i=0
input_ids = tokenizer(l, return_tensors='pt')
outputs = model.generate(**input_ids, 
                             max_length=64,
                             do_sample=True,
                             top_k=1,
                             num_return_sequences=10
                            )


print(f'sample {i+1} : {l} - {tokenizer.decode(outputs[0], skip_special_tokens=True)}.')
