import os
import ast
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions

load_dotenv()

# 1. Configuration
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    client = genai.Client(api_key=api_key)
    print("Gemini API Client initialized successfully.")
else:
    print("API Key not found. Check your .env file.")
    exit()

INPUT_CSV = "/home/rishabh/ThesisProject/DataCreation/merged_processed_output.csv"
OUTPUT_CSV = "generated_data/summarized_legal_docs.csv"

def parse_document(doc_str):
    try:
        return ast.literal_eval(doc_str)
    except:
        return []

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# 3. Load Data
df = pd.read_csv(INPUT_CSV)
results = []
doc_id = 0

def clean_to_paragraphs(text): # FIXED: Function defined
    text = re.sub(r'[#*_\-]', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

print(f"Starting summarization for {len(df)} documents using new google-genai SDK...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    doc_list = parse_document(row['document'])
    full_text = "".join(doc_list)
    
    if not full_text.strip():
        results.append({"doc_id": doc_id, "summary": "Empty Document"})
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
        response = client.models.generate_content(
            model="gemini-3-pro-preview", 
            contents=prompt
        )
        
        if response and response.text:
            final_summary = clean_to_paragraphs(response.text)
            results.append({"doc_id": doc_id, "summary": final_summary})
        
        time.sleep(2) 

    except exceptions.ResourceExhausted: # FIXED: Now recognized
        print(f"\n[!!!] API Limit Hit on Doc {doc_id}. Saving and exiting.")
        break 

    except Exception as e:
        print(f"\n[!] Skipping Doc {doc_id} due to: {e}")
        continue

# 4. Save
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Summaries saved to {OUTPUT_CSV}")