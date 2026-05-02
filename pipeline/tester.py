import pandas as pd
import numpy as np
from tqdm import tqdm

# --- MOCK DATA SECTION ---
# Replacing your 'load_data' and 'embed_data' dependencies with simple dictionaries

def mock_load_data():
    # Returns 2 examples for English (en) and Spanish (es)
    data = {
        'en': ["Hello, how are you?", "Close the door."],
        'es': ["Hola, ¿cómo estás?", "Cierra la puerta."]
    }
    labels = {
        'en': [4.5, 2.0],  # Mock politeness scores
        'es': [4.2, 1.8]
    }
    return data, labels

def mock_load_embeddings():
    # Returns random vectors to simulate BERT/LLM embeddings
    return {
        'en': np.random.rand(2, 8), # 2 entries, 8-dimension vectors
        'es': np.random.rand(2, 8)
    }

def mock_get_closest(target_data, num=2):
    # Just returns the first few examples instead of doing vector math
    return target_data[:num]

# --- SIMPLIFIED LOGIC ---

def align_embedding_simple(embedding, label, source_key, target_key, embeddings, labels):
    """
    Simplified math: Just adds a dummy 'difference' vector 
    to simulate moving from one language space to another.
    """
    source_mean = np.mean(embeddings[source_key], axis=0)
    target_mean = np.mean(embeddings[target_key], axis=0)
    diff = target_mean - source_mean
    return embedding + diff

def query_mock_llm(prompt):
    """
    The 'Heart' of the simplification: No LLM, just string manipulation.
    """
    return f"[MOCK TRANSLATION of: {prompt[:30]}...]"

def load_prompt_simple(source_key, target_key, text, label, rag_examples):
    # Hardcoded template instead of reading from a .txt file
    template = "Source ({}) to Target ({}). Text: '{}'. Style Level: {}. Examples: {}"
    return template.format(source_key, target_key, text, label, "|".join(rag_examples))

# --- THE MINI PIPELINE ---

def run_simple_pipeline():
    print("🚀 Starting Mock Pipeline...")
    
    # 1. Load Data
    data, labels = mock_load_data()
    embeddings = mock_load_embeddings()
    
    source_langs = ['en']
    target_langs = ['es']
    
    results = {}

    for src in source_langs:
        for tgt in target_langs:
            if src == tgt: continue
            
            print(f"\nProcessing {src} -> {tgt}")
            translations_list = []
            
            # 2. Iterate through data (only first 2 entries)
            for i in range(len(data[src])):
                text = data[src][i]
                label = labels[src][i]
                emb = embeddings[src][i]
                
                # 3. "Align" Embeddings
                aligned_emb = align_embedding_simple(emb, label, src, tgt, embeddings, labels)
                
                # 4. Get RAG Examples
                rag_exs = mock_get_closest(data[tgt])
                
                # 5. Build Prompt
                prompt = load_prompt_simple(src, tgt, text, label, rag_exs)
                
                # 6. "Query" Model
                translation = query_mock_llm(prompt)
                translations_list.append(translation)
                
                print(f"  Input: {text}")
                print(f"  Result: {translation}")

            results[f"{src}->{tgt}"] = translations_list

    # 7. Mock Save
    df_results = pd.DataFrame(results)
    print("\n--- Final Results Table ---")
    print(df_results)
    return df_results

if __name__ == "__main__":
    run_simple_pipeline()