import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import sys

while(1):
    file_path = input("Please enter the path to the code snippet file | Enter 'Exit' to terminate the program : ")
    
    if file_path.lower() == 'exit':
        sys.exit()    
    
    try:
        with open(file_path, 'r') as file:
            code_snippet = file.read()
            
        
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-code-classification-model')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-code-classification-model')

        model.eval()

        label_classes = np.load('label_classes.npy', allow_pickle=True)


        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_classes

        encoding = tokenizer.encode_plus(
            code_snippet,
            add_special_tokens=True,
            max_length=64, 
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()

        predicted_language = label_encoder.inverse_transform([predicted_label])[0]
        print(f"\n\nThe predicted programming language is: {predicted_language}\n\n")
            
    except FileNotFoundError:
        print(f"\nError: File '{file_path}' not found. Please check the file path and try again.\n")
