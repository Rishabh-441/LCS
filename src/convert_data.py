import os
import json
import requests
import zipfile
import io
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

# Configuration
URLS = {
    "train": "https://huggingface.co/datasets/opennyaiorg/InLegalNER/resolve/main/NER_TRAIN.zip",
    "dev": "https://huggingface.co/datasets/opennyaiorg/InLegalNER/resolve/main/NER_DEV.zip"
}

def download_and_extract_all(url, split_name):
    print(f"\nDownloading {split_name} data from {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return []

    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    combined_data = []
    
    # Iterate over ALL files in the zip
    for filename in z.namelist():
        if filename.endswith(".json"):
            print(f"  - Found file: {filename}")
            with z.open(filename) as f:
                try:
                    data = json.load(f)
                    # Check if data is a list (standard) or dict
                    if isinstance(data, list):
                        combined_data.extend(data)
                        print(f"    -> Added {len(data)} rows.")
                    else:
                        print(f"    -> Skipped {filename} (unexpected format: not a list)")
                except json.JSONDecodeError:
                    print(f"    -> Skipped {filename} (invalid JSON)")
    
    print(f"Total rows for {split_name}: {len(combined_data)}")
    return combined_data

def convert_to_spacy(json_data, output_file):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    skipped = 0
    
    print(f"Converting to {output_file}...")
    
    for item in tqdm(json_data):
        # OpenNyAI JSON structure usually looks like this:
        # { "data": { "text": "..." }, "annotations": [ { "result": [...] } ] }
        
        # 1. Get Text
        text = item.get('data', {}).get('text')
        if not text:
            # Fallback for flat structure
            text = item.get('text')
        
        if not text:
            continue
            
        doc = nlp.make_doc(text)
        ents = []
        
        # 2. Get Annotations
        annotations = item.get('annotations', [])
        if annotations:
            results = annotations[0].get('result', [])
        else:
            # Fallback for flat structure
            results = item.get('result', [])
            
        for res in results:
            val = res.get('value')
            if val:
                start = val.get('start')
                end = val.get('end')
                labels = val.get('labels')
                
                if start is not None and end is not None and labels:
                    span = doc.char_span(start, end, label=labels[0], alignment_mode="contract")
                    if span:
                        ents.append(span)
        
        try:
            doc.ents = spacy.util.filter_spans(ents)
            doc_bin.add(doc)
        except Exception:
            skipped += 1
            
    doc_bin.to_disk(output_file)
    print(f"Saved {output_file}. (Skipped {skipped} overlapping/invalid items)")

def main():
    # 1. Process Train (All files)
    train_data = download_and_extract_all(URLS["train"], "train")
    if train_data:
        convert_to_spacy(train_data, "train.spacy")
    
    # 2. Process Dev (All files)
    dev_data = download_and_extract_all(URLS["dev"], "dev")
    if dev_data:
        convert_to_spacy(dev_data, "dev.spacy")
    
    print("\nSUCCESS! You now have 'train.spacy' and 'dev.spacy' containing ALL data.")

if __name__ == "__main__":
    main()