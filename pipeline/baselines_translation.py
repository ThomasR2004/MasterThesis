import pandas as pd
from load_data import load_train_data, load_test_data, load_train_embeddings, load_test_embeddings
from embed_data import get_closest_embeddings
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import time
import random
from vllm import LLM, SamplingParams
import tiktoken


#List of all language keys for each style
TRANS_KEYS_POLITENESS = ['en->es', 'en->ja', 'en->zh', 'es->en', 'es->ja', 'es->zh', 'ja->en', 'ja->es', 'ja->zh', 'zh->en', 'zh->es', 'zh->ja']
TRANS_KEYS_INTIMACY = ['English->Spanish', 'English->Portuguese', 'English->Italian', 'English->French', 'English->Chinese', 'Spanish->English', 'Spanish->Portuguese', 'Spanish->Italian', 'Spanish->French', 'Spanish->Chinese', 'Portuguese->English', 'Portuguese->Spanish', 'Portuguese->Italian', 'Portuguese->French', 'Portuguese->Chinese', 'Italian->English', 'Italian->Spanish', 'Italian->Portuguese', 'Italian->French', 'Italian->Chinese', 'French->English', 'French->Spanish', 'French->Portuguese', 'French->Italian', 'French->Chinese', 'Chinese->English', 'Chinese->Spanish', 'Chinese->Portuguese', 'Chinese->Italian', 'Chinese->French']
TRANS_KEYS_FORMAL = ['en->fr', 'en->it', 'en->pt', 'fr->en', 'fr->it', 'fr->pt', 'it->en', 'it->fr', 'it->pt', 'pt->en', 'pt->fr', 'pt->it']

#List of all language keys for each style
LANG_KEYS_POLITENESS = ['en', 'es', 'ja', 'zh']
LANG_KEYS_INTIMACY = ['English', 'Spanish', 'Portuguese', 'Italian', 'French', 'Chinese']
LANG_KEYS_FORMAL = ['en', 'fr', 'it', 'pt']

def query_openai(prompt, model="gpt-4"):
    with open("API_KEY.txt", "r") as file:
        api_key = file.read()
    client = OpenAI(api_key=api_key)

    num_tries = 0
    for i in range(3):
        try:
            translation = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                model=model,
            )
            return translation.choices[0].message.content
        except Exception as e:
            num_tries += 1
            print("Try {}; Error: {}".format(str(num_tries), str(e)))     
            time.sleep(3)
    return "ERROR"


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


def load_direct_baseline_prompt(style, source_key, target_key, text):
    key_mapping = {"en": "English", "es": "Spanish", "ja": "Japanese", "zh": "Chinese", "fr": "French", "it": "Italian", "pt": "Portuguese"}

    if(style == "politeness"):
        source_key = key_mapping[source_key]
        target_key = key_mapping[target_key]
    elif(style == "formal"):
        source_key = key_mapping[source_key]
        target_key = key_mapping[target_key]
    
    prompt = "Translate the following text from {} to {}.\nText: {}\nOutput only the translation.".format(source_key, target_key, text)
    return prompt

def load_vanilla_baseline_prompt(style, source_key, target_key, text, label):
    key_mapping = {"en": "English", "es": "Spanish", "ja": "Japanese", "zh": "Chinese", "fr": "French", "it": "Italian", "pt": "Portuguese"}

    if(style == "politeness"):
        source_key = key_mapping[source_key]
        target_key = key_mapping[target_key]
        with open("prompts/politeness_baseline.txt", "r") as file:
            prompt = file.read()
    elif(style == "intimacy"):
        with open("prompts/intimacy_baseline.txt", "r") as file:
            prompt = file.read()
    elif(style == "formal"):
        source_key = key_mapping[source_key]
        target_key = key_mapping[target_key]
        with open("prompts/formality_baseline.txt", "r") as file:
            prompt = file.read() 
    
    prompt = prompt.format(source_key, target_key, text)
    return prompt

def load_fewshot_baseline_prompt(style, source_key, target_key, text, label, random_examples):
    key_mapping = {"en": "English", "es": "Spanish", "ja": "Japanese", "zh": "Chinese", "fr": "French", "it": "Italian", "pt": "Portuguese"}

    if(style == "politeness"):
        source_key = key_mapping[source_key]
        target_key = key_mapping[target_key]
        with open("prompts/politeness_baseline_fewshot.txt", "r") as file:
            prompt = file.read()
    elif(style == "intimacy"):
        with open("prompts/intimacy_baseline_fewshot.txt", "r") as file:
            prompt = file.read()
    elif(style == "formal"):
        source_key = key_mapping[source_key]
        target_key = key_mapping[target_key]
        with open("prompts/formality_baseline_fewshot.txt", "r") as file:
            prompt = file.read() 
    
    prompt = prompt.format(source_key, target_key, text, target_key, random_examples[0], random_examples[1], random_examples[2], random_examples[3], random_examples[4])
    return prompt


def get_politeness_scores(model_name):
    politeness_data_test, politeness_labels_test = load_test_data("politeness")
    politeness_data_train, politeness_labels_train = load_train_data("politeness")

    # 4 training examples of the different politeness levels
    politeness_examples = {}
    for key in LANG_KEYS_POLITENESS:
        politeness_examples[key] = []
        # get examples from 4 quantiles
        for i in range(4):
            quantile = np.quantile(politeness_labels_train[key], i/4)
            idx = np.argmin(np.abs(politeness_labels_train[key] - quantile))
            politeness_examples[key].append((politeness_data_train[key][idx], round(politeness_labels_train[key][idx], 3)))

    model = LLM(model_name, max_num_seqs=8, max_model_len=512, tensor_parallel_size=2, enforce_eager=True)
    params = SamplingParams(n=1, temperature=0.6, top_p=0.9, max_tokens=300)

    politeness_scores = {}
    for source_key in LANG_KEYS_POLITENESS:
        print("Scoring for language: ", source_key)
        politeness_scores[source_key] = []
        ex1, lab1 = politeness_examples[source_key][0]
        ex2, lab2 = politeness_examples[source_key][1]
        ex3, lab3 = politeness_examples[source_key][2]
        ex4, lab4 = politeness_examples[source_key][3]
        prompts = [f"Evaluate the politeness of the text on a range from 0-4.\nExample 1: {ex1}\nPoliteness: {lab1}\nExample 2: {ex2}\nPoliteness: {lab2}\nExample 3: {ex3}\nPoliteness: {lab3}\nExample 4: {ex4}\nPoliteness: {lab4}\nWhat is the politeness of the following text on a scale from 0-4? Output only the score at the end after \"Politeness:\".\n{sample}" \
                   for sample in politeness_data_test[source_key]]
        scores = query_vllm(prompts, model, params)
        parsed_scores = []
        for score in scores:
            try:
                parsed_score = float(score.split("Politeness: ")[1])
                parsed_scores.append(parsed_score)
            except:
                parsed_scores.append(2)
        politeness_scores[source_key] = parsed_scores
    
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    with open("translations/politeness_scores_{}.pkl".format(model_name), "wb") as file:
        pd.to_pickle(politeness_scores, file)


def vanilla_baseline_politeness(model_name="gpt-4"):
    politeness_data_test, politeness_labels_test = load_test_data("politeness")

    if not "gpt" in model_name:
        model = LLM(model_name, max_num_seqs=8, max_model_len=512, tensor_parallel_size=2, enforce_eager=True)
        params = SamplingParams(n=1, temperature=0.6, top_p=0.9, max_tokens=300)

    vanilla_baseline_dict = {}
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
                prompt = load_vanilla_baseline_prompt("politeness", source_key, target_key, utterance, round(label, 3))
                if "gpt" in model_name:
                    translation = query_openai(prompt, model=model_name)
                    print("{}: {}".format(i, translation))
                    translations_list.append(translation)
                else:
                    to_translate.append(prompt)
            if "gpt" not in model_name:
                translations_list = query_vllm(to_translate, model, params)
            vanilla_baseline_dict[trans_key] = translations_list  
    
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    with open("translations/vanilla_{}_politeness.pkl".format(model_name), "wb") as file:
        pd.to_pickle(vanilla_baseline_dict, file)

def fewshot_baseline_politeness(model_name="gpt-4"):
    politeness_data, politeness_labels = load_train_data("politeness")
    politeness_data_test, politeness_labels_test = load_test_data("politeness")

    fewshot_translations_dict = {}
    for source_key in LANG_KEYS_POLITENESS:
        for target_key in LANG_KEYS_POLITENESS:
            if(target_key == source_key): continue
            print("Translating from {} to {}".format(source_key, target_key))
            trans_key = "{}->{}".format(source_key, target_key)
            translations_list = []
            for i in tqdm(range(len(politeness_data_test[source_key]))):
                utterance = politeness_data_test[source_key][i]
                label = politeness_labels_test[source_key][i]
                random_examples = random.sample(politeness_data[target_key], 5)
                prompt = load_fewshot_baseline_prompt("politeness", source_key, target_key, utterance, round(label, 3), random_examples)
                translation = query_openai(prompt, model=model_name)
                print("{}: {}".format(i, translation))
                translations_list.append(translation)
            fewshot_translations_dict[trans_key] = translations_list
    
    with open("translations/fewshot_{}_politeness.pkl".format(model_name), "wb") as file:
        pd.to_pickle(fewshot_translations_dict, file)

def direct_baseline(style, keys, model_name="gpt-4"):
    intimacy_data_test, _ = load_test_data(style)

    if not "gpt" in model_name:
        model = LLM(model_name, max_num_seqs=8, max_model_len=512, tensor_parallel_size=2, enforce_eager=True)
        params = SamplingParams(n=1, temperature=0.6, top_p=0.9, max_tokens=300)

    vanilla_baseline_dict = {}
    for source_key in keys:
        for target_key in keys:
            if(target_key == source_key): continue
            print("Translating from {} to {}".format(source_key, target_key))
            trans_key = "{}->{}".format(source_key, target_key)
            translations_list = []
            to_translate = []
            for i in tqdm(range(len(intimacy_data_test[source_key]))):
                utterance = intimacy_data_test[source_key][i]
                prompt = load_direct_baseline_prompt(style, source_key, target_key, utterance)
                if "gpt" in model_name:
                    translation = query_openai(prompt, model=model_name)
                    print("{}: {}".format(i, translation))
                    translations_list.append(translation)
                else:
                    to_translate.append(prompt)
            if "gpt" not in model_name:
                translations_list = query_vllm(to_translate, model, params)
            vanilla_baseline_dict[trans_key] = translations_list
    
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    with open("translations/direct_{}_{}.pkl".format(model_name, style), "wb") as file:
        pd.to_pickle(vanilla_baseline_dict, file)

def estimate_cost_direct_baseline(style, keys, model_name="gpt-4"):
    intimacy_data_test, _ = load_test_data(style)

    assert "gpt" in model_name, "Only OpenAI models are supported for cost estimation"

    vanilla_baseline_dict = {}
    for source_key in keys:
        for target_key in keys:
            if(target_key == source_key): continue
            print("Translating from {} to {}".format(source_key, target_key))
            num_input_tokens = 0
            num_output_tokens = 0
            for i in tqdm(range(len(intimacy_data_test[source_key]))):
                utterance = intimacy_data_test[source_key][i]
                prompt = load_direct_baseline_prompt(style, source_key, target_key, utterance)
                num_input_tokens += tiktoken.count(prompt)
                translation = query_openai(prompt, model=model_name)
                num_output_tokens += len(translation.split())


def vanilla_baseline_intimacy(model_name="gpt-4"):
    intimacy_data_test, intimacy_labels_test = load_test_data("intimacy")

    if not "gpt" in model_name:
        model = LLM(model_name, max_num_seqs=8, max_model_len=512, tensor_parallel_size=2, enforce_eager=True)
        params = SamplingParams(n=1, temperature=0.6, top_p=0.9, max_tokens=300)

    vanilla_baseline_dict = {}
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
                prompt = load_vanilla_baseline_prompt("intimacy", source_key, target_key, utterance, round(label, 3))
                if "gpt" in model_name:
                    translation = query_openai(prompt, model=model_name)
                    print("{}: {}".format(i, translation))
                    translations_list.append(translation)
                else:
                    to_translate.append(prompt)
            if "gpt" not in model_name:
                translations_list = query_vllm(to_translate, model, params)
            vanilla_baseline_dict[trans_key] = translations_list
    
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    with open("translations/vanilla_{}_intimacy.pkl".format(model_name), "wb") as file:
        pd.to_pickle(vanilla_baseline_dict, file)

def fewshot_baseline_intimacy(model_name="gpt-4"):
    intimacy_data, intimacy_labels = load_train_data("intimacy")
    intimacy_data_test, intimacy_labels_test = load_test_data("intimacy")

    fewshot_translations_dict = {}
    for source_key in LANG_KEYS_INTIMACY:
        for target_key in LANG_KEYS_INTIMACY:
            if(target_key == source_key): continue
            print("Translating from {} to {}".format(source_key, target_key))
            trans_key = "{}->{}".format(source_key, target_key)
            translations_list = []
            for i in tqdm(range(len(intimacy_data_test[source_key]))):
                utterance = intimacy_data_test[source_key][i]
                label = intimacy_labels_test[source_key][i]
                random_examples = random.sample(intimacy_data[target_key], 5)
                prompt = load_fewshot_baseline_prompt("intimacy", source_key, target_key, utterance, round(label, 3), random_examples)
                translation = query_openai(prompt, model=model_name)
                print("{}: {}".format(i, translation))
                translations_list.append(translation)
            fewshot_translations_dict[trans_key] = translations_list
    
    with open("translations/fewshot_{}_intimacy.pkl".format(model_name), "wb") as file:
        pd.to_pickle(fewshot_translations_dict, file)

def vanilla_baseline_formal(model_name="gpt-4"):
    formal_data_test, formal_labels_test = load_test_data("formal")

    if not "gpt" in model_name:
        model = LLM(model_name, max_num_seqs=8, max_model_len=512, tensor_parallel_size=2, enforce_eager=True)
        params = SamplingParams(n=1, temperature=0.6, top_p=0.9, max_tokens=300)

    vanilla_baseline_dict = {}
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
                prompt = load_vanilla_baseline_prompt("formal", source_key, target_key, utterance, round(label, 3))
                if "gpt" in model_name:
                    translation = query_openai(prompt, model=model_name)
                    print("{}: {}".format(i, translation))
                    translations_list.append(translation)
                else:
                    to_translate.append(prompt)
            if "gpt" not in model_name:
                translations_list = query_vllm(to_translate, model, params)
            vanilla_baseline_dict[trans_key] = translations_list
    
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    with open("translations/vanilla_{}_formal.pkl".format(model_name), "wb") as file:
        pd.to_pickle(vanilla_baseline_dict, file)

def fewshot_baseline_formal(model_name="gpt-4"):
    formal_data, formal_labels = load_train_data("formal")
    formal_data_test, formal_labels_test = load_test_data("formal")

    fewshot_translations_dict = {}
    for source_key in LANG_KEYS_FORMAL:
        for target_key in LANG_KEYS_FORMAL:
            if(target_key == source_key): continue
            print("Translating from {} to {}".format(source_key, target_key))
            trans_key = "{}->{}".format(source_key, target_key)
            translations_list = []
            for i in tqdm(range(len(formal_data_test[source_key]))):
                utterance = formal_data_test[source_key][i]
                label = formal_labels_test[source_key][i]
                random_examples = random.sample(formal_data[target_key], 5)
                prompt = load_fewshot_baseline_prompt("formal", source_key, target_key, utterance, round(label, 3), random_examples)
                translation = query_openai(prompt, model=model_name)
                print("{}: {}".format(i, translation))
                translations_list.append(translation)
            fewshot_translations_dict[trans_key] = translations_list
    
    with open("translations/fewshot_{}_formal.pkl".format(model_name), "wb") as file:
        pd.to_pickle(fewshot_translations_dict, file)

def redo_errors_politeness(model_name="gpt-4"):
    politeness_data, politeness_labels = load_train_data("politeness")
    politeness_data_test, politeness_labels_test = load_test_data("politeness")

    with open("translations/fewshot_{}_politeness.pkl".format(model_name), "rb") as file:
        politeness_translations = pd.read_pickle(file)
    for source_key in LANG_KEYS_POLITENESS:
        for target_key in LANG_KEYS_POLITENESS:
            if(target_key == source_key): continue
            trans_key = "{}->{}".format(source_key, target_key)
            translations_list = politeness_translations[trans_key]
            for i in range(len(translations_list)):
                if(translations_list[i] == "ERROR"):
                    print("Translating utterance {} from {} to {}".format(i, source_key, target_key))
                    utterance = politeness_data_test[source_key][i]
                    label = politeness_labels_test[source_key][i]
                    random_examples = random.sample(politeness_data[target_key], 5)
                    prompt = load_fewshot_baseline_prompt("politeness", source_key, target_key, utterance, round(label, 3), random_examples)
                    translation = query_openai(prompt, model=model_name)
                    print("{}: {}".format(i, translation))
                    translations_list[i] = translation
            politeness_translations[trans_key] = translations_list
    
    with open ("translations/fewshot_{}_politeness_redo.pkl".format(model_name), "wb") as file:
        pd.to_pickle(politeness_translations, file)


if __name__ == "__main__":
    # fewshot_baseline_politeness()
    # vanilla_baseline_politeness("meta-llama/Llama-3.2-11B-Vision-Instruct")
    # get_politeness_scores("meta-llama/Llama-3.2-11B-Vision-Instruct")
    # fewshot_baseline_intimacy()
    # vanilla_baseline_intimacy("meta-llama/Llama-3.2-11B-Vision-Instruct")
    # fewshot_baseline_formal()
    # vanilla_baseline_formal("meta-llama/Llama-3.2-11B-Vision-Instruct")

    direct_baseline("politeness", LANG_KEYS_POLITENESS, "gpt-4")

    # direct_baseline("formal", LANG_KEYS_FORMAL, "meta-llama/Llama-3.2-11B-Vision-Instruct")
    # direct_baseline("intimacy", LANG_KEYS_INTIMACY, "meta-llama/Llama-3.2-11B-Vision-Instruct")
    # direct_baseline("politeness", LANG_KEYS_POLITENESS, "meta-llama/Llama-3.2-11B-Vision-Instruct")

    # direct_baseline("formal", LANG_KEYS_FORMAL, "gpt-3.5-turbo-0125")
    # direct_baseline("intimacy", LANG_KEYS_INTIMACY, "gpt-3.5-turbo-0125")
    # direct_baseline("politeness", LANG_KEYS_POLITENESS, "gpt-3.5-turbo-0125")
