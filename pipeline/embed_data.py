import pandas as pd
from load_data import load_train_data, load_test_data
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine

#List of all language keys for each style
LANG_KEYS_POLITENESS = ['en', 'es', 'ja', 'zh']
LANG_KEYS_INTIMACY = ['English', 'Spanish', 'Portuguese', 'Italian', 'French', 'Chinese']
LANG_KEYS_FORMAL = ['en', 'fr', 'it', 'pt']

def embed_data(data, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_name = "nvidia/NV-Embed-v2"
    model_name = "BAAI/bge-m3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    embeddings = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # batch_embeddings = outputs["sentence_embeddings"].mean(dim=1).cpu().numpy()
        batch_embeddings = outputs["pooler_output"].cpu().numpy()
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    return embeddings

def run_embed(split = "train"):
    if(split == "train"):
        politeness_data, politeness_labels = load_train_data("politeness")
        intimacy_data, intimacy_labels = load_train_data("intimacy")
        formality_data, formality_labels = load_train_data("formal")
    elif(split == "test"):
        politeness_data, politeness_labels = load_test_data("politeness")
        intimacy_data, intimacy_labels = load_test_data("intimacy")
        formality_data, formality_labels = load_test_data("formal")

    politeness_data_embeddings = {}
    for lang_key in LANG_KEYS_POLITENESS:
        print("Embedding politeness data for language: {}".format(lang_key))
        politeness_data_embeddings[lang_key] = embed_data(politeness_data[lang_key])
        assert(len(politeness_data[lang_key]) == len(politeness_data_embeddings[lang_key]))
    with open("data/{}_data/politeness_embeddings_m3.pkl".format(split), "wb") as f:
        pd.to_pickle(politeness_data_embeddings, f)
    
    intimacy_data_embeddings = {}
    for lang_key in LANG_KEYS_INTIMACY:
        print("Embedding intimacy data for language: {}".format(lang_key))
        intimacy_data_embeddings[lang_key] = embed_data(intimacy_data[lang_key])
        assert(len(intimacy_data[lang_key]) == len(intimacy_data_embeddings[lang_key]))
    with open("data/{}_data/intimacy_embeddings_m3.pkl".format(split), "wb") as f:
        pd.to_pickle(intimacy_data_embeddings, f)
    
    formality_data_embeddings = {}
    for lang_key in LANG_KEYS_FORMAL:
        print("Embedding formality data for language: {}".format(lang_key))
        formality_data_embeddings[lang_key] = embed_data(formality_data[lang_key])
        assert(len(formality_data[lang_key]) == len(formality_data_embeddings[lang_key]))
    with open("data/{}_data/formal_embeddings_m3.pkl".format(split), "wb") as f:
        pd.to_pickle(formality_data_embeddings, f)

#Fcn to get the N closest embeddings to a given embedding, within a certain label range
def get_closest_embeddings(style, source_data, source_embeddings, source_labels, aligned_embedding, aligned_label, N):
    if(style == "politeness"): 
        label_range = 0.333
    elif(style == "intimacy"):
        label_range = 0.333
    elif(style == "formal"):
        label_range = 0
    else:
        raise ValueError("Invalid style")
    
    cosine_similarities = [1 - cosine(aligned_embedding, emb) for emb in source_embeddings]

    # Sort the indices from most similar to least
    sorted_indices = np.argsort(cosine_similarities)[::-1]

    #select the top N indices that are within label_range of label
    top_n_indices = []
    for i in sorted_indices:
        if abs(source_labels[i] - aligned_label) <= label_range:
            top_n_indices.append(i)
        if len(top_n_indices) == N:
            break

    return [source_data[i] for i in top_n_indices], [cosine_similarities[i] for i in top_n_indices], [source_labels[i] for i in top_n_indices]

if __name__ == "__main__":
    run_embed("train")
    run_embed("test")