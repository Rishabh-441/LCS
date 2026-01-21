import os
import ast
import re
import pandas as pd
from tqdm import tqdm
import torch
from dotenv import load_dotenv
from transformers import pipeline, BitsAndBytesConfig

torch.cuda.empty_cache()

load_dotenv()

# 1. Configuration
INPUT_CSV = "/home/rishabh/ThesisProject/DataCreation/merged_processed_output.csv"
OUTPUT_CSV = "generated_data/summarized_legal_docs_llama_8b.csv"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

hf_token = os.getenv("HF_TOKEN")

# 2. Define the Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Initialize Pipeline
print(f"Loading {MODEL_ID}... (Local Inference)")
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    device_map="auto",
    token=hf_token,
    model_kwargs={
        "dtype": torch.bfloat16,
        "quantization_config": quant_config
    }
)

def parse_document(doc_str):
    try:
        return ast.literal_eval(doc_str)
    except:
        return []

def clean_to_paragraphs(text):
    text = re.sub(r'[#*_\-]', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# 3. Load Data
df = pd.read_csv(INPUT_CSV)
results = []
doc_id = 0 

print(f"Starting summarization for {len(df)} documents using Llama-3.1-8B...")

# 4. Processing Loop
for index, row in tqdm(df.iterrows(), total=len(df)):
    doc_list = parse_document(row['document'])
    full_text = "".join(doc_list)
    
    if not full_text.strip():
        results.append({"doc_id": doc_id, "summary": "Empty Document"})
        doc_id += 1
        continue
    
    prompt = f"""
    You are a legal expert tasked with producing a clear, comprehensive, and faithful summary of the following legal document.

    Write the summary in well-formed, coherent paragraphs using formal legal language.
    Capture the core facts of the case, the legal issues involved, the arguments or positions of the parties, the reasoning applied, and the final outcome or holding, where applicable.
    Preserve logical flow and causal relationships between events and decisions.
    Avoid paraphrasing so aggressively that legal meaning, nuance, or intent is lost.

    Do not include headings, titles, bullet points, numbering, or any form of markdown.
    Do not add commentary, opinions, or external information.
    Do not quote the document verbatim unless essential for legal accuracy.
    Return only the plain text of the summary paragraphs.

    Document:
    "{full_text}"
    
    Summary :
    """

    try:
        messages = [{"role": "user", "content": prompt}]
        
        outputs = pipe(
            messages,
            max_new_tokens=2048,
            temperature=0.2,
            do_sample=True,
        )
        
        generated_content = outputs[0]["generated_text"][-1]["content"]
        
        if generated_content:
            final_summary = clean_to_paragraphs(generated_content)
            results.append({"doc_id": doc_id, "summary": final_summary})
            
        doc_id += 1

        # CHECKPOINT LOGIC: Every 1000 summaries
        if len(results) % 1000 == 0 and len(results) > 0:
            checkpoint_df = pd.DataFrame(results)
            # Mode 'a' for append, header=False if file exists to prevent duplicate headers
            file_exists = os.path.isfile(OUTPUT_CSV)
            checkpoint_df.to_csv(OUTPUT_CSV, mode='a', index=False, header=not file_exists)
            print(f"\n[Checkpoint] {len(results)} summaries saved to {OUTPUT_CSV}")
            results = [] # Clear memory of processed list after saving

    except torch.cuda.OutOfMemoryError:
        print(f"\n[!!!] CUDA Out of Memory on Doc {doc_id}. Saving remaining and exiting.")
        break
    except Exception as e:
        print(f"\n[!] Skipping Doc {doc_id} due to: {e}")
        doc_id += 1
        continue

# 5. Final Save (for the remaining records < 1000)
if results:
    final_df = pd.DataFrame(results)
    file_exists = os.path.isfile(OUTPUT_CSV)
    final_df.to_csv(OUTPUT_CSV, mode='a', index=False, header=not file_exists)

print(f"Summarization complete. Final data saved to {OUTPUT_CSV}")