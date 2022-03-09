import torch
from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-large")

# For transformers v4.x+:
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large", use_fast=False)

# For transformers v3.x:
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
sentence = 'Chúng_tôi là những nghiên_cứu_viên .'

input_ids = torch.tensor([tokenizer.encode(sentence)])

with torch.no_grad():
    features = phobert(input_ids)  # Models outputs are now tuples

text = tokenizer(input_ids)
print(features)
## With TensorFlow 2.0+:
# from transformers import TFAutoModel
# phobert = TFAutoModel.from_pretrained("vinai/phobert-base")