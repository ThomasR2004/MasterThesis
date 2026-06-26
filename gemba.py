import openai
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


def gemba(source_lang, target_lang, source_segs, target_segs):
    with open("API_KEY.txt", "r") as file:
        api_key = file.read()
    client = openai.OpenAI(api_key=api_key)

    prompt = """Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100, where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar". Output just the score as an integer and nothing else.

{source_lang} source: "{source_seg}"
{target_lang} translation: "{target_seg}"
"""

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def eval_model(src, tgt, src_text, tgt_text):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=3,
            messages=[
                {"role": "user", "content": prompt.format(source_lang=src, target_lang=tgt, source_seg=src_text, target_seg=tgt_text)},
            ]
        )
        return completion.choices[0].message.content

    scores = []
    for src, tgt in tqdm(zip(source_segs, target_segs)):
        s = eval_model(source_lang, target_lang, src, tgt)
        print(s)
        if s.isnumeric():
            scores.append(int(s))
        else:
            scores.append(0)

    return scores