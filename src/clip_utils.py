import os

import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection


def load_model():
    os.environ['HF_HOME'] = '/viscam/u/mattzh1314'
    return CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", cache_dir='/viscam/u/mattzh1314')

def load_tokenizer():
    os.environ['HF_HOME'] = '/viscam/u/mattzh1314'
    return AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir='/viscam/u/mattzh1314')

def get_text_embed(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.text_embeds.squeeze(0)

def get_similarity(e1, e2):
    dot_product = torch.dot(e1, e2)
    norm_e1 = torch.norm(e1)
    norm_e2 = torch.norm(e2)
    similarity = dot_product / (norm_e1 * norm_e2)
    return similarity.item()

def get_average_similarity(e1_arr, e2_arr):
    e1_batch = torch.stack(e1_arr)
    e2_batch = torch.stack(e2_arr)
    
    e1_norm = torch.nn.functional.normalize(e1_batch, p=2, dim=1)
    e2_norm = torch.nn.functional.normalize(e2_batch, p=2, dim=1)
    
    similarity_matrix = torch.mm(e1_norm, e2_norm.t())
    
    average_similarity = similarity_matrix.mean().item()
    
    return average_similarity
