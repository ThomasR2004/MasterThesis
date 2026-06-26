from tqdm import tqdm
from comet import download_model, load_from_checkpoint


def cometkiwi(source_segs, target_segs, model=None):
    # model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
    if model is None:
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        model = load_from_checkpoint(model_path)

    data = [{"src": src, "mt": tgt} for src, tgt in zip(source_segs, target_segs)]
    model_output = model.predict(data, batch_size=8, gpus=1)
    return model_output.scores


if __name__ == "__main__":
    source_segs = [
        "This is a test.",
        "I am a student.",
        "The weather is nice.",
    ]
    target_segs = [
        "C'est un poisson.",
        "Je suis poisson",
        "Il fait poisson.",
    ]
    cometkiwi(source_segs, target_segs)