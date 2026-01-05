# import json
# import csv
# import re
# import dateparser
# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm  # <--- Added import

# # ==========================================
# # 1. SETUP BERT MODEL (Global Load)
# # ==========================================

# print("Loading InLegalBERT model... (This may take a moment)")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Use your local path if needed, otherwise "law-ai/InLegalBERT"
# model_name = "law-ai/InLegalBERT" 
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name).to(device)

# # Threshold for Semantic Similarity
# SIMILARITY_THRESHOLD = 0.8

# # ==========================================
# # 2. HELPER FUNCTIONS
# # ==========================================

# def normalize_text(text):
#     """
#     Light normalization:
#     - Lowercase
#     - Strip leading/trailing whitespace
#     - KEEPS spaces and special characters inside the string
#     """
#     if not isinstance(text, str): return ""
#     return text.lower().strip()

# def get_parsed_dates(ner_data):
#     """
#     Extracts entities labeled 'DATE', parses them, and returns a list of valid datetime objects.
#     """
#     dt_list = []
#     if not ner_data or "DATE" not in ner_data:
#         return dt_list
    
#     for date_str in ner_data["DATE"]:
#         try:
#             dt = dateparser.parse(date_str)
#             if dt:
#                 dt_list.append(dt)
#         except:
#             continue
#     return dt_list

# def get_flat_entity_set(ner_data):
#     """
#     Flattens entities for exact string matching.
#     """
#     entity_set = set()
#     if not ner_data:
#         return entity_set
        
#     for label, entities in ner_data.items():
#         for entity in entities:
#             norm_entity = normalize_text(entity)
#             if norm_entity: 
#                 entity_set.add(norm_entity)
#     return entity_set

# def get_all_raw_entities(ner_data):
#     """
#     Returns a simple list of ALL raw entity strings from the source (for BERT comparison).
#     """
#     raw_list = []
#     if not ner_data:
#         return raw_list
#     for label, entities in ner_data.items():
#         raw_list.extend(entities)
#     return raw_list

# # --- BERT FUNCTIONS ---

# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output.last_hidden_state
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
#     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#     return sum_embeddings / sum_mask

# def check_semantic_match_bert(summary_text, source_texts):
#     """
#     Compares 'summary_text' against a LIST of 'source_texts' efficiently.
#     Returns True if ANY source text has similarity > SIMILARITY_THRESHOLD.
#     """
#     if not source_texts:
#         return False
    
#     encoded_summ = tokenizer([summary_text], padding=True, truncation=True, return_tensors='pt').to(device)
    
#     with torch.no_grad():
#         out_summ = model(**encoded_summ)
#         emb_summ = mean_pooling(out_summ, encoded_summ['attention_mask'])
#         emb_summ = F.normalize(emb_summ, p=2, dim=1)

#     # Iterate through source texts
#     for src_text in source_texts:
#         encoded_src = tokenizer([src_text], padding=True, truncation=True, return_tensors='pt').to(device)
        
#         with torch.no_grad():
#             out_src = model(**encoded_src)
#             emb_src = mean_pooling(out_src, encoded_src['attention_mask'])
#             emb_src = F.normalize(emb_src, p=2, dim=1)
        
#         score = F.cosine_similarity(emb_summ, emb_src).item()
        
#         if score >= SIMILARITY_THRESHOLD:
#             return True 
            
#     return False

# # --- DATE FUNCTION ---

# def is_date_match(summary_date_str, source_dt_objects):
#     try:
#         summ_dt = dateparser.parse(summary_date_str)
#         if summ_dt is None:
#             return False
#         for src_dt in source_dt_objects:
#             if summ_dt.date() == src_dt.date():
#                 return True
#     except:
#         return False
#     return False

# def calculate_lhi_details(source_ner, summary_ner):
#     """
#     LHI Calculation with Date, Exact (Light Norm), and BERT matching.
#     """
#     E_src_norm = get_flat_entity_set(source_ner) # For Exact Match
#     src_dt_objs = get_parsed_dates(source_ner)   # For Date Match
#     src_raw_list = get_all_raw_entities(source_ner) # For BERT Match

#     summary_entities_flat = []
#     if summary_ner:
#         for label, entities in summary_ner.items():
#             for ent in entities:
#                 summary_entities_flat.append((ent, label))

#     summary_count = len(summary_entities_flat)
#     if summary_count == 0:
#         return "N/A", 0, 0, []

#     hallucinated_entities = []
    
#     for raw_text, label in summary_entities_flat:
#         is_match = False
        
#         # A. Semantic Date Matching
#         if label == "DATE":
#             if is_date_match(raw_text, src_dt_objs):
#                 is_match = True
        
#         # B. Exact String Matching (Now includes spaces/symbols)
#         if not is_match:
#             norm_text = normalize_text(raw_text)
#             if norm_text in E_src_norm:
#                 is_match = True
                
#         # C. InLegalBERT Semantic Similarity
#         if not is_match:
#             if check_semantic_match_bert(raw_text, src_raw_list):
#                 is_match = True
        
#         if not is_match:
#             hallucinated_entities.append(raw_text)

#     hallucination_count = len(hallucinated_entities)
#     lhi_score = 1 - (hallucination_count / summary_count)
    
#     return lhi_score, summary_count, hallucination_count, hallucinated_entities

# # ==========================================
# # 3. MAIN EXECUTION
# # ==========================================

# SOURCE_FILE = 'row_wise_legal_entities.json'
# SUMMARY_FILE = 'row_wise_legal_entities_summ.json'
# OUTPUT_CSV = 'lhi_results_bert_lightnorm.csv'

# print("Loading data files...")

# try:
#     with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
#         source_data = json.load(f)
        
#     with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
#         summary_data = json.load(f)

#     print(f"Data loaded. Processing {len(summary_data)} documents...")

#     all_scores = []

#     with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(['Doc_ID', 'LHI_Score', 'Total_Summary_Entities', 'Hallucination_Count', 'Hallucinated_Entities'])
        
#         # Wrapped loop in tqdm for progress bar
#         for doc_id, summary_ner in tqdm(summary_data.items(), desc="Processing Documents"):
#             source_ner = source_data.get(doc_id, {})
            
#             score, total, h_count, h_list = calculate_lhi_details(source_ner, summary_ner)
            
#             if score != "N/A":
#                 all_scores.append(score)
            
#             h_list_str = "; ".join(h_list)
#             csv_writer.writerow([doc_id, score, total, h_count, h_list_str])

#     print(f"\nDone! Results saved to '{OUTPUT_CSV}'")
    
#     if all_scores:
#         avg_lhi = sum(all_scores) / len(all_scores)
#         print("-" * 30)
#         print(f"Global Average LHI Score: {avg_lhi:.4f}")
#         print("-" * 30)
#     else:
#         print("No valid scores calculated.")

# except FileNotFoundError as e:
#     print(f"Error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")






# ____________________
import json
import csv
import re
import dateparser
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ==========================================
# 1. SETUP BERT MODEL (Global Load)
# ==========================================

print("Loading InLegalBERT model... (This may take a moment)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use your local path if needed, otherwise "law-ai/InLegalBERT"
model_name = "law-ai/InLegalBERT" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Threshold for Semantic Similarity
SIMILARITY_THRESHOLD = 0.8

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def normalize_text(text):
    """
    Light normalization:
    - Lowercase
    - Strip leading/trailing whitespace
    - KEEPS spaces and special characters inside the string
    """
    if not isinstance(text, str): return ""
    return text.lower().strip()

def get_parsed_dates(ner_data):
    """
    Extracts entities labeled 'DATE', parses them, and returns a list of valid datetime objects.
    """
    dt_list = []
    if not ner_data or "DATE" not in ner_data:
        return dt_list
    
    for date_str in ner_data["DATE"]:
        try:
            dt = dateparser.parse(date_str)
            if dt:
                dt_list.append(dt)
        except:
            continue
    return dt_list

def get_flat_entity_set(ner_data):
    """
    Flattens entities for exact string matching.
    """
    entity_set = set()
    if not ner_data:
        return entity_set
        
    for label, entities in ner_data.items():
        for entity in entities:
            norm_entity = normalize_text(entity)
            if norm_entity: 
                entity_set.add(norm_entity)
    return entity_set

def get_all_raw_entities(ner_data):
    """
    Returns a simple list of ALL raw entity strings from the source (for BERT comparison).
    """
    raw_list = []
    if not ner_data:
        return raw_list
    for label, entities in ner_data.items():
        raw_list.extend(entities)
    return raw_list

# --- BERT FUNCTIONS ---

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def check_semantic_match_bert(summary_text, source_texts):
    """
    Compares 'summary_text' against a LIST of 'source_texts' efficiently.
    Returns True if ANY source text has similarity > SIMILARITY_THRESHOLD.
    """
    if not source_texts:
        return False
    
    encoded_summ = tokenizer([summary_text], padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        out_summ = model(**encoded_summ)
        emb_summ = mean_pooling(out_summ, encoded_summ['attention_mask'])
        emb_summ = F.normalize(emb_summ, p=2, dim=1)

    # Iterate through source texts
    for src_text in source_texts:
        encoded_src = tokenizer([src_text], padding=True, truncation=True, return_tensors='pt').to(device)
        
        with torch.no_grad():
            out_src = model(**encoded_src)
            emb_src = mean_pooling(out_src, encoded_src['attention_mask'])
            emb_src = F.normalize(emb_src, p=2, dim=1)
        
        score = F.cosine_similarity(emb_summ, emb_src).item()
        
        if score >= SIMILARITY_THRESHOLD:
            return True 
            
    return False

# --- DATE FUNCTION ---

def is_date_match(summary_date_str, source_dt_objects):
    try:
        summ_dt = dateparser.parse(summary_date_str)
        if summ_dt is None:
            return False
        for src_dt in source_dt_objects:
            if summ_dt.date() == src_dt.date():
                return True
    except:
        return False
    return False

def calculate_lhi_details(source_ner, summary_ner):
    """
    LHI Calculation with Date, Exact (Light Norm), and BERT matching.
    """
    E_src_norm = get_flat_entity_set(source_ner) # For Exact Match
    src_dt_objs = get_parsed_dates(source_ner)   # For Date Match
    src_raw_list = get_all_raw_entities(source_ner) # For BERT Match

    summary_entities_flat = []
    if summary_ner:
        for label, entities in summary_ner.items():
            for ent in entities:
                summary_entities_flat.append((ent, label))

    summary_count = len(summary_entities_flat)
    if summary_count == 0:
        return "N/A", 0, 0, []

    hallucinated_entities = []
    
    for raw_text, label in summary_entities_flat:
        is_match = False
        
        # A. Semantic Date Matching
        if label == "DATE":
            if is_date_match(raw_text, src_dt_objs):
                is_match = True
        
        # B. Exact String Matching (Now includes spaces/symbols)
        if not is_match:
            norm_text = normalize_text(raw_text)
            if norm_text in E_src_norm:
                is_match = True
                
        # C. InLegalBERT Semantic Similarity
        if not is_match:
            if check_semantic_match_bert(raw_text, src_raw_list):
                is_match = True
        
        if not is_match:
            hallucinated_entities.append(raw_text)

    hallucination_count = len(hallucinated_entities)
    lhi_score = 1 - (hallucination_count / summary_count)
    
    return lhi_score, summary_count, hallucination_count, hallucinated_entities

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

SOURCE_FILE = 'row_wise_legal_entities.json'
SUMMARY_FILE = 'row_wise_legal_entities_summ.json'
OUTPUT_CSV = 'lhi_results_trial_20.csv'

print("Loading data files...")

try:
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
        
    with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)

    print(f"Data loaded. Selecting FIRST 20 documents for TRIAL...")

    # Select only the first 20 items for the trial run
    trial_items = list(summary_data.items())[:20]

    all_scores = []

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Doc_ID', 'LHI_Score', 'Total_Summary_Entities', 'Hallucination_Count', 'Hallucinated_Entities'])
        
        # Iterate over the sliced trial data
        for doc_id, summary_ner in tqdm(trial_items, desc="Running Trial (20 Docs)"):
            source_ner = source_data.get(doc_id, {})
            
            score, total, h_count, h_list = calculate_lhi_details(source_ner, summary_ner)
            
            if score != "N/A":
                all_scores.append(score)
            
            h_list_str = "; ".join(h_list)
            csv_writer.writerow([doc_id, score, total, h_count, h_list_str])

    print(f"\nDone! Results saved to '{OUTPUT_CSV}'")
    
    if all_scores:
        avg_lhi = sum(all_scores) / len(all_scores)
        print("-" * 30)
        print(f"Trial Average LHI Score: {avg_lhi:.4f}")
        print("-" * 30)
    else:
        print("No valid scores calculated in this trial.")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")