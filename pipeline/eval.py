from load_data import load_test_data, load_test_embeddings, load_train_data
import argparse
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
from gemba import gemba
from cometkiwi import cometkiwi
import os
import pandas as pd
import torch
from transformers import BitsAndBytesConfig
from datasets import Dataset
from comet import download_model, load_from_checkpoint


def our_metric(source, target):
    # source_norm = np.array(source) / np.max(source)
    # target_norm = np.array(target) / np.max(target)
    # return 1 - np.sqrt(np.mean(np.square(source_norm - target_norm)))
    return pearsonr(source, target).correlation


def get_style_model(style, lang, dir="../"):
    # pipe = pipeline("text-classification", model=f"../training/{lang}_{style}", tokenizer='FacebookAI/xlm-roberta-large', device="cuda", function_to_apply="none")
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(f"{dir}training/{lang}/{style}_lora", num_labels=1, quantization_config=config, device_map="auto")
    pipe = pipeline("text-classification", model=model, tokenizer='mistralai/Mistral-7B-v0.1', function_to_apply="none")
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id
    return pipe


def load_formal_data(dir="../"):
    languages = ["en", "fr", "it", "pt"]

    data = {lang: ([], []) for lang in languages}
    labels = {lang: [] for lang in languages}
    df_en = pd.read_csv(dir + "data/labeled_data/formal/en.csv")
    data["en"] = df_en["text"].tolist()
    labels["en"] = df_en["label"].tolist()

    for lang in languages[1:]:
        df = pd.read_csv(dir + f"/data/labeled_data/formal/{lang}.csv")
        data[lang] = df["text"].tolist()
        labels[lang] = df["label"].tolist()
    return data, labels


def main(args):
    data_test, labels_test = load_test_data(args.style)

    if args.method == "direct" and "Llama" not in args.model:
        translations = pickle.load(open(f"../data/{args.model}_{args.style}_translated_data.pkl", "rb"))
        if args.style == "formal":
            data_test, labels_test = load_formal_data()
        else:
            data_test, labels_test = load_train_data(args.style)
    else:
        translations = pickle.load(open(f"translations/{args.method}_{args.model}_{args.style}.pkl", "rb"))

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    translated_labels = {}
    all_gemba = {}
    all_cometkiwi = {}

    # model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    # comet_model = load_from_checkpoint(model_path)

    total_tokens = 0
    for key in list(translations.keys()):
        print(f"Evaluating {key}")

        # style labels
        style_labels_t = []
        style_labels_s = []

        # get max length
        print(f"Max length: {max([len(translations[key][i]) for i in range(len(translations[key]))])}")
        print(f"Min length: {min([len(translations[key][i]) for i in range(len(translations[key]))])}")
        
        print(np.sum([1 if "ERROR" in translations[key][i] else 0 for i in range(len(translations[key]))]))

        # truncate translations after newline
        translations[key] = [translations[key][i].split("\n")[0] for i in range(len(translations[key]))]


        lang1 = key.split("->")[0]
        lang2 = key.split("->")[1]

        # if args.style == "intimacy":
        #     lens = np.array([len(tokenizer(data_test[lang1][i])["input_ids"]) for i in range(len(data_test[lang1]))])
        #     mask = lens > 15
        #     labels_test[lang1] = np.array(labels_test[lang1])[mask]
        #     data_test[lang1] = [data_test[lang1][i] for i in range(len(data_test[lang1])) if mask[i]]
        #     translations[key] = [translations[key][i] for i in range(len(translations[key])) if mask[i]]

        # indices = np.arange(len(labels_test[lang1]))

        # model = get_style_model(args.style, lang2)
        # print("Loaded style model")
        # data = [translations[key][i] for i in indices]
        # for out in tqdm(model(data, batch_size=8), total=len(translations[key])):
        #     style_labels_t.append(out['score'])

        # del model
        # torch.cuda.empty_cache()

        # style_labels_s = np.array(labels_test[lang1])
        # style_labels_t = np.array(style_labels_t)
        # style_labels_s = np.array(style_labels_s)

        # translated_labels[key] = style_labels_t

        # # corr = pearsonr(style_labels_s, style_labels_t).correlation

        # # cometkiwi
        # assert len(data_test[lang1]) == len(translations[key])
        # cometkiwi_scores = cometkiwi(data_test[lang1], translations[key], model=comet_model)
        # cometkiwi_score = np.mean(cometkiwi_scores)

        # gemba
        gemba_scores = gemba(lang1, lang2, data_test[lang1], translations[key])
        # print("Total tokens:", sum(gemba_scores))
        gemba_score = np.mean(gemba_scores)

        # # results[key] = {"corr": corr, "cometkiwi": cometkiwi_score}
        all_gemba[key] = gemba_scores
        # all_cometkiwi[key] = cometkiwi_scores

        # # write results to txt csv file
        # with open(f"results/{args.model}_{args.style}.txt", "a") as f:
        #     f.write(f"{args.model},{args.method},{key},{corr},{cometkiwi_score},{gemba_score}\n")

    # pickle.dump(translated_labels, open(f"../data/{args.model}_{args.method}_{args.style}_labels.pkl", "wb"))
    pickle.dump(all_gemba, open(f"../data/{args.model}_{args.method}_{args.style}_gemba.pkl", "wb"))
    # pickle.dump(all_cometkiwi, open(f"../data/{args.model}_{args.method}_{args.style}_cometkiwi.pkl", "wb"))


def fix(args):
    data_test, labels_test = load_test_data(args.style)

    if args.method == "direct":
        translations = pickle.load(open(f"../data/{args.model}_{args.style}_translated_data.pkl", "rb"))
        if args.style == "formal":
            data_test, labels_test = load_formal_data()
        else:
            data_test, labels_test = load_train_data(args.style)
    else:
        translations = pickle.load(open(f"translations/{args.method}_{args.model}_{args.style}.pkl", "rb"))

    translated_labels = pickle.load(open(f"../data/{args.model}_{args.method}_{args.style}_labels.pkl", "rb"))
    all_gemba = pickle.load(open(f"../data/{args.model}_{args.method}_{args.style}_gemba.pkl", "rb"))
    for key in list(translations.keys()):
        print(f"Evaluating {key}")

        # get max length
        print(f"Max length: {max([len(translations[key][i]) for i in range(len(translations[key]))])}")
        print(f"Min length: {min([len(translations[key][i]) for i in range(len(translations[key]))])}")
        
        print(np.sum([1 if "ERROR" in translations[key][i] else 0 for i in range(len(translations[key]))]))

        # truncate translations after newline
        orig_translations = translations[key]
        translations[key] = [translations[key][i].split("\n")[0] for i in range(len(translations[key]))]
        style_labels_t = translated_labels[key]

        lang1 = key.split("->")[0]
        lang2 = key.split("->")[1]

        indices = np.nonzero(np.array(orig_translations) != np.array(translations[key]))[0]
        model = get_style_model(args.style, lang2)
        print("Loaded style model")
        data = [translations[key][i] for i in indices]
        print("Number to fix:", len(data))
        for i, out in enumerate(tqdm(model(data, batch_size=8), total=len(translations[key]))):
            style_labels_t[indices[i]] = out['score']
        cometkiwi_scores = cometkiwi(data_test[lang1], translations[key])
        cometkiwi_score = np.mean(cometkiwi_scores)

        corr = pearsonr(labels_test[lang1], style_labels_t).correlation

        gemba_scores_new = gemba(lang1, lang2, [data_test[lang1][i] for i in indices], data)
        all_gemba[key] = np.array(all_gemba[key])
        all_gemba[key][indices] = gemba_scores_new
        gemba_score = np.mean(all_gemba[key])

        print(args.model, args.method, key, f"Correlation: {corr}", f"COMET-Kiwi: {cometkiwi_score}", f"GEMBA: {gemba_score}")

        with open(f"results/{args.model}_{args.style}.txt", "a") as f:
            f.write(f"{args.model},{args.method},{key},{corr},{cometkiwi_score},{gemba_score}\n")

    pickle.dump(translated_labels, open(f"../data/{args.model}_{args.method}_{args.style}_labels.pkl", "wb"))
    pickle.dump(all_gemba, open(f"../data/{args.model}_{args.method}_{args.style}_gemba.pkl", "wb"))


def create_tables(args):
    if os.path.exists(f"../data/{args.model}_rasta_{args.style}_gemba.pkl"):
        corr_res = pickle.load(open(f"../data/{args.model}_rasta_{args.style}_labels.pkl", "rb"))
        comet_res = pickle.load(open(f"../data/{args.model}_rasta_{args.style}_cometkiwi.pkl", "rb"))
        gemba_res = pickle.load(open(f"../data/{args.model}_rasta_{args.style}_gemba.pkl", "rb"))
        corr_vanilla = pickle.load(open(f"../data/{args.model}_vanilla_{args.style}_labels.pkl", "rb"))
        comet_vanilla = pickle.load(open(f"../data/{args.model}_vanilla_{args.style}_cometkiwi.pkl", "rb"))
        gemba_vanilla = pickle.load(open(f"../data/{args.model}_vanilla_{args.style}_gemba.pkl", "rb"))
        corr_fewshot = pickle.load(open(f"../data/{args.model}_fewshot_{args.style}_labels.pkl", "rb"))
        comet_fewshot = pickle.load(open(f"../data/{args.model}_fewshot_{args.style}_cometkiwi.pkl", "rb"))
        gemba_fewshot = pickle.load(open(f"../data/{args.model}_fewshot_{args.style}_gemba.pkl", "rb"))
        if not os.path.exists(f"../data/{args.model}_direct_{args.style}_gemba.pkl"):
            print("Missing:", f"../data/{args.model}_direct_{args.style}_gemba.pkl")
            # set corr_base to corr_res with all 0
            _, labels_train = load_train_data(args.style)
            corr_base = {key: [0] * len(labels_train[key.split("->")[0]]) for key in corr_res}
            comet_base = {key: [0] * len(labels_train[key.split("->")[0]]) for key in corr_res}
            gemba_base = {key: [0] * len(labels_train[key.split("->")[0]]) for key in corr_res}
        else:
            corr_base = pickle.load(open(f"../data/{args.model}_direct_{args.style}_labels.pkl", "rb"))
            comet_base = pickle.load(open(f"../data/{args.model}_direct_{args.style}_cometkiwi.pkl", "rb"))
            gemba_base = pickle.load(open(f"../data/{args.model}_direct_{args.style}_gemba.pkl", "rb"))
    else:
        print("Missing:", f"../data/{args.model}_rasta_{args.style}_gemba.pkl")
        return

    _, labels_test = load_test_data(args.style)
    if args.style == "formal" and not "Llama" in args.model:
        _, labels_train = load_formal_data()
    elif "Llama" in args.model:
        _, labels_train = load_test_data(args.style)
    else:
        _, labels_train = load_train_data(args.style)

    corrs = {}
    comets = {}
    gembas = {}
    for key in list(corr_res.keys()):
        target_lang = key.split("->")[1]
        if target_lang not in corrs:
            corrs[target_lang] = {"rasta": [], "vanilla": [], "direct": [], "fewshot": []}
            comets[target_lang] = {"rasta": [], "vanilla": [], "direct": [], "fewshot": []}
            gembas[target_lang] = {"rasta": [], "vanilla": [], "direct": [], "fewshot": []}
        corrs[target_lang]["rasta"].append(our_metric(corr_res[key], labels_test[key.split("->")[0]]))
        comets[target_lang]["rasta"].append(np.mean(comet_res[key]))
        gembas[target_lang]["rasta"].append(np.mean(gemba_res[key]))

        corrs[target_lang]["vanilla"].append(our_metric(corr_vanilla[key], labels_test[key.split("->")[0]]))
        comets[target_lang]["vanilla"].append(np.mean(comet_vanilla[key]))
        gembas[target_lang]["vanilla"].append(np.mean(gemba_vanilla[key]))

        corrs[target_lang]["direct"].append(our_metric(corr_base[key], labels_train[key.split("->")[0]]))
        comets[target_lang]["direct"].append(np.mean(comet_base[key]))
        gembas[target_lang]["direct"].append(np.mean(gemba_base[key]))

        corrs[target_lang]["fewshot"].append(our_metric(corr_fewshot[key], labels_test[key.split("->")[0]]))
        comets[target_lang]["fewshot"].append(np.mean(comet_fewshot[key]))
        gembas[target_lang]["fewshot"].append(np.mean(gemba_fewshot[key]))
    
    # with open(f"results/{args.model}_{args.style}.txt", "r") as f:
    #     lines = f.readlines()
    #     corrs = {}
    #     comets = {}
    #     for line in lines:
    #         if line == "\n":
    #             continue
    #         model, method, key, corr, cometkiwi, gemba = line.split(",")
    #         target_lang = key.split("->")[1]
    #         if target_lang not in corrs:
    #             corrs[target_lang] = {}
    #             comets[target_lang] = {}
    #         if model not in corrs[target_lang]:
    #             corrs[target_lang][model] = {}
    #             comets[target_lang][model] = {}
    #         if method not in corrs[target_lang][model]:
    #             corrs[target_lang][model][method] = []
    #             comets[target_lang][model][method] = []
    #         corrs[target_lang][model][method].append(float(corr))
    #         comets[target_lang][model][method].append(float(cometkiwi))

    # lang2name = {"en": "English", "fr": "French", "it": "Italian", "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese", "es": "Spanish", "Chinese": "Chinese", "Japanese": "Japanese", "Spanish": "Spanish", "English": "English", "French": "French", "Italian": "Italian", "Portuguese": "Portuguese"}
    lang2name = {"en": "En", "fr": "Fr", "it": "It", "pt": "Pt", "zh": "Zh", "ja": "Ja", "es": "Es", "Chinese": "Zh", "Japanese": "Ja", "Spanish": "Es", "English": "En", "French": "Fr", "Italian": "It", "Portuguese": "Pt"}
    if max([len(l) for l in corrs.keys()]) > 2:
        languages = sorted(list(corrs.keys()))[::-1]
    else:
        languages = sorted(list(corrs.keys()))
    language_names = [lang2name[lang] for lang in languages]

    print(language_names)

    baseline = "direct"

    delta_corr = (np.mean(corrs[languages[0]]['rasta']) - np.mean(corrs[languages[0]][baseline])) / np.mean(corrs[languages[0]][baseline]) * 100
    delta_comet = (np.mean(comets[languages[0]]['rasta']) - np.mean(comets[languages[0]][baseline])) / np.mean(comets[languages[0]][baseline]) * 100
    delta_gemba = (np.mean(gembas[languages[0]]['rasta']) - np.mean(gembas[languages[0]][baseline])) / np.mean(gembas[languages[0]][baseline]) * 100
    # print(f"\\multirow{{14}}{{*}}{{{args.model}}} & \\multirow{{{len(corrs.keys())}}}{{*}}{{{args.style}}} & {language_names[0]} & {np.mean(corrs[languages[0]][baseline]):.2f} & {np.mean(comets[languages[0]][baseline]):.2f} & {np.mean(gembas[languages[0]][baseline]):.2f} & {np.mean(corrs[languages[0]]['vanilla']):.2f} & {np.mean(comets[languages[0]]['vanilla']):.2f} & {np.mean(gembas[languages[0]]['vanilla']):.2f} & {np.mean(corrs[languages[0]]['rasta']):.2f} & {np.mean(comets[languages[0]]['rasta']):.2f} & {np.mean(gembas[languages[0]]['rasta']):.2f} \\\\")
    print(f"\\multirow{{14}}{{*}}{{{args.model}}} & \\multirow{{{len(corrs.keys())}}}{{*}}{{{args.style}}} & {language_names[0]} & {np.mean(corrs[languages[0]]['fewshot']):.2f} & {np.mean(comets[languages[0]]['fewshot']):.2f} & {np.mean(gembas[languages[0]]['fewshot']):.2f}\\\\")
    for lang in languages[1:]:
        delta_corr = (np.mean(corrs[lang]['rasta']) - np.mean(corrs[lang][baseline])) / np.mean(corrs[lang][baseline]) * 100
        delta_comet = (np.mean(comets[lang]['rasta']) - np.mean(comets[lang][baseline])) / np.mean(comets[lang][baseline]) * 100
        delta_gemba = (np.mean(gembas[lang]['rasta']) - np.mean(gembas[lang][baseline])) / np.mean(gembas[lang][baseline]) * 100
        # print(f"& & {lang2name[lang]} & {np.mean(corrs[lang][baseline]):.2f} & {np.mean(comets[lang][baseline]):.2f} & {np.mean(gembas[lang][baseline]):.2f} & {np.mean(corrs[lang]['vanilla']):.2f} & {np.mean(comets[lang]['vanilla']):.2f} & {np.mean(gembas[lang]['vanilla']):.2f} & {np.mean(corrs[lang]['rasta']):.2f} & {np.mean(comets[lang]['rasta']):.2f} & {np.mean(gembas[lang]['rasta']):.2f} \\\\")
        print(f"{np.mean(corrs[lang]['fewshot']):.2f} & {np.mean(comets[lang]['fewshot']):.2f} & {np.mean(gembas[lang]['fewshot']):.2f}\\\\")
    # print(f" & & Avg. & {np.mean([np.mean(corrs[lang][baseline]) for lang in languages]):.2f} & {np.mean([np.mean(comets[lang][baseline]) for lang in languages]):.2f} & {np.mean([np.mean(gembas[lang][baseline]) for lang in languages]):.2f} & {np.mean([np.mean(corrs[lang]['vanilla']) for lang in languages]):.2f} & {np.mean([np.mean(comets[lang]['vanilla']) for lang in languages]):.2f} & {np.mean([np.mean(gembas[lang]['vanilla']) for lang in languages]):.2f} & {np.mean([np.mean(corrs[lang]['rasta']) for lang in languages]):.2f} & {np.mean([np.mean(comets[lang]['rasta']) for lang in languages]):.2f} & {np.mean([np.mean(gembas[lang]['rasta']) for lang in languages]):.2f} \\\\")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4", choices=["google", "gpt-4", "gpt-3.5", "Llama-3.2-11B-Vision-Instruct", "nllb", "gemma"])
    parser.add_argument("--style", type=str, default="all", choices=["politeness", "intimacy", "formal", "all"])
    parser.add_argument("--method", type=str, default="all", choices=["fewshot", "vanilla", "rasta", "direct", "all"])
    parser.add_argument("--create_tables", action="store_true")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    if args.style == "all":
        styles = ["intimacy", "formal", "politeness"]
    else:
        styles = [args.style]
    if args.method == "all":
        methods = ["direct", "vanilla", "rasta"]
    else:
        methods = [args.method]

    if args.create_tables:
        for style in styles:
            args.style = style
            create_tables(args)
    else:
        for style in styles:
            for method in methods:
                args.style = style
                args.method = method
                if args.fix:
                    fix(args)
                else:
                    main(args)