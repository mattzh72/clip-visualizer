import os

import torch
import clip


def load_model():
    model, preprocess = clip.load("ViT-B/32", device='cuda', download_root='/viscam/u/mattzh1314/')
    return model

def get_text_embed(sentence, model):
    text = clip.tokenize([sentence]).cuda()

    with torch.no_grad():
        text_features = model.encode_text(text)
    
    return text_features.squeeze(0)
    
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
