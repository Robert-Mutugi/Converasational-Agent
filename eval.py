import gzip
import shutil
import torch
import tensorflow as tf
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AdamW
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


'''with gzip.open('data_validation.jsonl.gz') as f:
    with open('eval.jsonl', 'wb') as i:
        shutil.copyfileobj(f, i)'''
file = pd.read_json(path_or_buf='training_set.jsonl', lines=True)

df = file['utterances']
x = []
y = []

i=0
ignore= "Hi, I'm your automated assistant."
for r in df[0:50]:
    for f in r:
        if f.find(ignore) == 0 : continue
        if i % 2 == 0: x.append(f)
        if i % 2 != 0: y.append(f)
        i = i+1
        
list(x)
list(y)

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-large-msmarco')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-large-msmarco')
model.to(device)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
xtrain_encode = tokenizer(x_train, truncation=True, padding=True)
xtest_encode = tokenizer(x_test, truncation=True, padding=True)
ytrain_encode = tokenizer(y_train, truncation=True, padding=True)
ytest_encode = tokenizer(y_test, truncation=True, padding=True)

#metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1) 
    acc = accuracy_score(y_true=labels, y_pred=logits)
    recall = recall_score(y_true=labels, y_pred=logits)
    precision = precision_score(y_true=labels, y_pred=logits)
    f1 = f1_score(y_true=labels, y_pred=logits)
    return {'predictions': predictions, 'precision': precision, 'f1':f1, 'acc':acc, 'recall': recall, 'labels': labels}


class ForT5Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        input_ids = torch.tensor(self.inputs["input_ids"][index]).squeeze()
        target_ids = torch.tensor(self.targets["input_ids"][index]).squeeze()
        
        return {"input_ids": input_ids, "labels": target_ids}

training_set = ForT5Dataset(xtrain_encode, ytrain_encode)
validation_set = ForT5Dataset(xtest_encode, ytest_encode)

# Define Trainer
args = TrainingArguments(
    output_dir="final_output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    save_steps=3000,
    load_best_model_at_end = True,    
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=training_set,
    eval_dataset=validation_set,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

#Train Model
#trainer.train()
#Save model
#trainer.save_model('ORQuAC_model')
model = T5ForConditionalGeneration.from_pretrained('ORQuAC_model')
#cm = confusion_matrix(y_test, output)

#Predict
#f= open('trained_predictions.txt','x')
file = open('trained_predictions.txt', 'w')

tokenize = tokenizer(x, padding=True, truncation=True)
encode = DataLoader(tokenize)
for i in range(100):
    input_ids = tokenizer.encode(x[i], return_tensors='pt').to(device)
    predictions = model.generate(input_ids=input_ids, max_length=64, do_sample=True, top_k=1, num_return_sequences=10)
    #print(predictions)

    for f in range(1):
        y_pred = tokenizer.decode(predictions[f], skip_special_tokens=True)
        print(y_pred)
        file.write(y_pred)
        file.write('\n')

file.close()








'''DataSet(xtest_encode)


train_args = {
    'output_dir': './trainresults',
    'evaluate_during_training': True,
    'max_seq_length': 128,
    'num_train_epochs': 4,
    'evaluate_during_trainig_steps':1000,
    'train_batch_size': 128,
    'eval_batch_size': 64
}

model.train()

opt = AdamW(model.parameters(), lr=5e-5)

val_loader = DataLoader(train_data, batch_size=16, shuffle=True)
acc=[]

loop = tqdm(train_data)
for batch in loop:
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)



'''