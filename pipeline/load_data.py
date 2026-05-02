import openai
import pandas as pd

def load_train_data(style_name, dir=""):
    train_data_filepath = dir + "data/train_data/{}_data.pkl".format(style_name)
    train_labels_filepath = dir + "data/train_data/{}_labels.pkl".format(style_name)

    train_data = pd.read_pickle(train_data_filepath)
    train_labels = pd.read_pickle(train_labels_filepath)

    return train_data, train_labels

def load_test_data(style_name, dir=""):
    test_data_filepath = dir + "data/test_data/{}_data.pkl".format(style_name)
    test_labels_filepath = dir + "data/test_data/{}_labels.pkl".format(style_name)

    test_data = pd.read_pickle(test_data_filepath)
    test_labels = pd.read_pickle(test_labels_filepath)

    return test_data, test_labels

def load_train_embeddings(style_name):
    embeddings_filepath = "data/train_data/{}_embeddings_m3.pkl".format(style_name)
    with open(embeddings_filepath, "rb") as f:
        embeddings = pd.read_pickle(f)
    return embeddings

def load_test_embeddings(style_name):
    embeddings_filepath = "data/test_data/{}_embeddings_m3.pkl".format(style_name)
    with open(embeddings_filepath, "rb") as f:
        embeddings = pd.read_pickle(f)
    return embeddings


def load_translations(model_name, style_name):
    translations_filepath = "translations/vanilla_{}_{}.pkl".format(model_name, style_name)
    with open(translations_filepath, "rb") as f:
        translations = pd.read_pickle(f)
    print(translations["en->zh"])

if __name__ == "__main__":
    #get length of all train data
    politeness_data, politeness_labels = load_train_data("politeness")
    for lang_key in politeness_data.keys():
        print(lang_key, len(politeness_data[lang_key]))
    
    intimacy_data, intimacy_labels = load_train_data("intimacy")
    for lang_key in intimacy_data.keys():
        print(lang_key, len(intimacy_data[lang_key]))
    
    formality_data, formality_labels = load_train_data("formal")
    for lang_key in formality_data.keys():
        print(lang_key, len(formality_data[lang_key]))

