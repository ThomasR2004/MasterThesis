import pandas as pd
from load_data import load_train_data, load_test_data, load_train_embeddings, load_test_embeddings
from embed_data import get_closest_embeddings
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import time
from vllm import LLM, SamplingParams

#List of all language keys for each style
TRANS_KEYS_POLITENESS = ['en->es', 'en->ja', 'en->zh', 'es->en', 'es->ja', 'es->zh', 'ja->en', 'ja->es', 'ja->zh', 'zh->en', 'zh->es', 'zh->ja']
TRANS_KEYS_INTIMACY = ['English->Spanish', 'English->Portuguese', 'English->Italian', 'English->French', 'English->Chinese', 'Spanish->English', 'Spanish->Portuguese', 'Spanish->Italian', 'Spanish->French', 'Spanish->Chinese', 'Portuguese->English', 'Portuguese->Spanish', 'Portuguese->Italian', 'Portuguese->French', 'Portuguese->Chinese', 'Italian->English', 'Italian->Spanish', 'Italian->Portuguese', 'Italian->French', 'Italian->Chinese', 'French->English', 'French->Spanish', 'French->Portuguese', 'French->Italian', 'French->Chinese', 'Chinese->English', 'Chinese->Spanish', 'Chinese->Portuguese', 'Chinese->Italian', 'Chinese->French']
TRANS_KEYS_FORMAL = ['en->fr', 'en->it', 'en->pt', 'fr->en', 'fr->it', 'fr->pt', 'it->en', 'it->fr', 'it->pt', 'pt->en', 'pt->fr', 'pt->it']

#List of all language keys for each style
LANG_KEYS_POLITENESS = ['en', 'es', 'ja', 'zh']
LANG_KEYS_INTIMACY = ['English', 'Spanish', 'Portuguese', 'Italian', 'French', 'Chinese']
LANG_KEYS_FORMAL = ['en', 'fr', 'it', 'pt']



def query_vllm(prompts, model, params):
    # model = LLM(model_name, max_num_seqs=1, tensor_parallel_size=2, max_model_len=4096)
    # params = SamplingParams(n=1, temperature=0.6, top_p=0.9)
    # print the longest prompt
    outputs = model.chat(
        [[{
            "role": "user",
            "content": prompt,
        }] for prompt in prompts],
        sampling_params=params
    )
    translations = [output.outputs[0].text for output in outputs]
    return translations


def align_embedding(style, embedding, label, source_key, target_key, embeddings, labels):
    # embeddings = load_train_embeddings(style)
    source_embeddings = embeddings[source_key]
    target_embeddings = embeddings[target_key]

    # labels = load_train_data(style)[1]
    source_labels = labels[source_key]
    target_labels = labels[target_key]
    
    # if(style == "politeness"):
    #     label_range = 1.667
    # elif(style == "intimacy"):
    #     label_range = 1.667
    # elif(style == "formal"):
    #     label_range = 0
    
    # source_indexes = [i for i, source_label in enumerate(source_labels) if abs(source_label - label) <= label_range]
    # target_indexes = [i for i, target_label in enumerate(target_labels) if abs(target_label - label) <= label_range]
    
    source_diffs = [abs(source_label - label) for source_label in source_labels]
    target_diffs = [abs(target_label - label) for target_label in target_labels]

    #sort source and target embeddings by difference from label and select the top 10%
    num_embs_source = int(len(source_embeddings)*0.1)
    num_embs_target = int(len(target_embeddings)*0.1)
    if(style == "formality"): 
        num_embs_source = np.count_nonzero(source_diffs == 0)
        num_embs_target = np.count_nonzero(target_diffs == 0)

    source_indexes = np.argsort(source_diffs)[:num_embs_source]
    target_indexes = np.argsort(target_diffs)[:num_embs_target]
    
    source_embeddings_mean = np.mean(source_embeddings[source_indexes], axis=0)
    target_embeddings_mean = np.mean(target_embeddings[target_indexes], axis=0)
    
    diff = target_embeddings_mean - source_embeddings_mean
    return (embedding + diff)

def load_prompt(style, source_key, target_key, text, label, rag_examples):
    key_mapping = {"en": "English", "es": "Spanish", "ja": "Japanese", "zh": "Chinese", "fr": "French", "it": "Italian", "pt": "Portuguese"}

    if(style == "politeness"):
        source_key = key_mapping[source_key]
        target_key = key_mapping[target_key]
        with open("prompts/politeness_translation.txt", "r") as file:
            prompt = file.read()
    elif(style == "intimacy"):
        with open("prompts/intimacy_translation.txt", "r") as file:
            prompt = file.read()
    elif(style == "formal"):
        source_key = key_mapping[source_key]
        target_key = key_mapping[target_key]
        with open("prompts/formality_translation.txt", "r") as file:
            prompt = file.read() 
    
    prompt = prompt.format(source_key, target_key, text, label, source_key, target_key, rag_examples[0], rag_examples[1], rag_examples[2], rag_examples[3], rag_examples[4], label, target_key)
    return prompt


def rasta_translation_politeness(model_name="gpt-4"):
    politeness_data, politeness_labels = load_train_data("politeness")
    politeness_embeddings = load_train_embeddings("politeness")

    politeness_data_test, politeness_labels_test = load_test_data("politeness")
    politeness_embeddings_test = load_test_embeddings("politeness")

    if not "gpt" in model_name:
        model = LLM(model_name, max_num_seqs=8, max_model_len=1024, tensor_parallel_size=2, enforce_eager=True)
        params = SamplingParams(n=1, temperature=0.6, top_p=0.9, max_tokens=300)

    rasta_translations_dict = {}
    for source_key in LANG_KEYS_POLITENESS:
        for target_key in LANG_KEYS_POLITENESS:
            if(target_key == source_key): continue
            print("Translating from {} to {}".format(source_key, target_key))
            trans_key = "{}->{}".format(source_key, target_key)
            translations_list = []
            to_translate = []
            for i in tqdm(range(len(politeness_data_test[source_key]))):
                utterance = politeness_data_test[source_key][i]
                label = politeness_labels_test[source_key][i]
                embedding = politeness_embeddings_test[source_key][i]
                aligned_embedding = align_embedding("politeness", embedding, label, source_key, target_key, politeness_embeddings, politeness_labels)
                rag_examples, _, _ = get_closest_embeddings("politeness", politeness_data[target_key], politeness_embeddings[target_key], politeness_labels[target_key], aligned_embedding, label, 5)
                prompt = load_prompt("politeness", source_key, target_key, utterance, round(label, 3), rag_examples)
                if "gpt" in model_name:
                    translation = query_openai(prompt, model=model_name)
                    print("{}: {}".format(i, translation))
                    translations_list.append(translation)
                else:
                    to_translate.append(prompt)
            if "gpt" not in model_name:
                translations_list = query_vllm(to_translate, model, params)
                print(translations_list)
            rasta_translations_dict[trans_key] = translations_list   

    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    with open("translations/rasta_{}_politeness.pkl".format(model_name), "wb") as file:
        pd.to_pickle(rasta_translations_dict, file)

def rasta_translation_intimacy(model_name="gpt-4"):
    intimacy_data, intimacy_labels = load_train_data("intimacy")
    intimacy_embeddings = load_train_embeddings("intimacy")

    intimacy_data_test, intimacy_labels_test = load_test_data("intimacy")
    intimacy_embeddings_test = load_test_embeddings("intimacy")

    if not "gpt" in model_name:
        model = LLM(model_name, max_num_seqs=8, max_model_len=1024, tensor_parallel_size=2, enforce_eager=True)
        params = SamplingParams(n=1, temperature=0.6, top_p=0.9, max_tokens=300)

    rasta_translations_dict = {}
    for source_key in LANG_KEYS_INTIMACY:
        for target_key in LANG_KEYS_INTIMACY:
            if(target_key == source_key): continue
            print("Translating from {} to {}".format(source_key, target_key))
            trans_key = "{}->{}".format(source_key, target_key)
            translations_list = []
            to_translate = []
            for i in tqdm(range(len(intimacy_data_test[source_key]))):
                utterance = intimacy_data_test[source_key][i]
                label = intimacy_labels_test[source_key][i]
                embedding = intimacy_embeddings_test[source_key][i]
                aligned_embedding = align_embedding("intimacy", embedding, label, source_key, target_key, intimacy_embeddings, intimacy_labels)
                rag_examples, _, _ = get_closest_embeddings("intimacy", intimacy_data[target_key], intimacy_embeddings[target_key], intimacy_labels[target_key], aligned_embedding, label, 5)
                prompt = load_prompt("intimacy", source_key, target_key, utterance, round(label, 3), rag_examples)
                if "gpt" in model_name:
                    translation = query_openai(prompt, model=model_name)
                    print("{}: {}".format(i, translation))
                    translations_list.append(translation)
                else:
                    to_translate.append(prompt)
    
            if "gpt" not in model_name:
                translations_list = query_vllm(to_translate, model, params)
                print(translations_list)
            rasta_translations_dict[trans_key] = translations_list   

    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    with open("translations/rasta_{}_intimacy.pkl".format(model_name), "wb") as file:
        pd.to_pickle(rasta_translations_dict, file)

def rasta_translation_formal(model_name="gpt-4"):
    formal_data, formal_labels = load_train_data("formal")
    formal_embeddings = load_train_embeddings("formal")

    formal_data_test, formal_labels_test = load_test_data("formal")
    formal_embeddings_test = load_test_embeddings("formal")

    if not "gpt" in model_name:
        model = LLM(model_name, max_num_seqs=8, max_model_len=1024, tensor_parallel_size=2, enforce_eager=True)
        params = SamplingParams(n=1, temperature=0.6, top_p=0.9, max_tokens=300)

    rasta_translations_dict = {}
    for source_key in LANG_KEYS_FORMAL:
        for target_key in LANG_KEYS_FORMAL:
            if(target_key == source_key): continue
            print("Translating from {} to {}".format(source_key, target_key))
            trans_key = "{}->{}".format(source_key, target_key)
            translations_list = []
            to_translate = []
            for i in tqdm(range(len(formal_data_test[source_key]))):
                utterance = formal_data_test[source_key][i]
                label = formal_labels_test[source_key][i]
                embedding = formal_embeddings_test[source_key][i]
                aligned_embedding = align_embedding("formal", embedding, label, source_key, target_key, formal_embeddings, formal_labels)
                rag_examples, _, _ = get_closest_embeddings("formal", formal_data[target_key], formal_embeddings[target_key], formal_labels[target_key], aligned_embedding, label, 5)
                prompt = load_prompt("formal", source_key, target_key, utterance, round(label, 3), rag_examples)
                if "gpt" in model_name:
                    translation = query_openai(prompt, model=model_name)
                    print("{}: {}".format(i, translation))
                    translations_list.append(translation)
                else:
                    to_translate.append(prompt)
            if "gpt" not in model_name:
                translations_list = query_vllm(to_translate, model, params)
                print(translations_list)
            rasta_translations_dict[trans_key] = translations_list   

    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    with open("translations/rasta_{}_formal.pkl".format(model_name), "wb") as file:
        pd.to_pickle(rasta_translations_dict, file)

def redo_errors_politeness(model_name="gpt-4"):
    politeness_data, politeness_labels = load_train_data("politeness")
    politeness_embeddings = load_train_embeddings("politeness")

    politeness_data_test, politeness_labels_test = load_test_data("politeness")
    politeness_embeddings_test = load_test_embeddings("politeness")
    with open("translations/rasta_{}_politeness.pkl".format(model_name), "rb") as file:
        politeness_translations = pd.read_pickle(file)

    for source_key in LANG_KEYS_POLITENESS:
        for target_key in LANG_KEYS_POLITENESS:
            if(target_key == source_key): continue
            trans_key = "{}->{}".format(source_key, target_key)
            translations_list = politeness_translations[trans_key]
            for i in range(len(translations_list)):
                if(translations_list[i] == "ERROR"):
                    print("Redoing translation {} from {} to {}".format(i, source_key, target_key))
                    utterance = politeness_data_test[source_key][i]
                    label = politeness_labels_test[source_key][i]
                    embedding = politeness_embeddings_test[source_key][i]
                    aligned_embedding = align_embedding("politeness", embedding, label, source_key, target_key)
                    rag_examples, _, _ = get_closest_embeddings("politeness", politeness_data[target_key], politeness_embeddings[target_key], politeness_labels[target_key], aligned_embedding, label, 5)
                    prompt = load_prompt("politeness", source_key, target_key, utterance, round(label, 3), rag_examples)
                    translation = query_openai(prompt, model=model_name)
                    print("{}: {}".format(i, translation))
                    translations_list[i] = translation
            politeness_translations[trans_key] = translations_list
    
    with open("translations/rasta_{}_politeness_redo.pkl".format(model_name), "wb") as file:
        pd.to_pickle(politeness_translations, file)


if __name__ == "__main__":
    # rasta_translation_politeness("meta-llama/Llama-3.2-11B-Vision-Instruct")
    # rasta_translation_formal("meta-llama/Llama-3.2-11B-Vision-Instruct")
    rasta_translation_intimacy("meta-llama/Llama-3.2-11B-Vision-Instruct")
    # rasta_translation_politeness()
    # rasta_translation_intimacy()
    # rasta_translation_formal()
    # redo_errors_politeness()
