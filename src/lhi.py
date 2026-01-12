# import json
# import csv
# import re
# import dateparser
# from tqdm import tqdm
# from thefuzz import fuzz  # <--- Added fuzzy matching

# # ==========================================
# # 1. HELPER FUNCTIONS
# # ==========================================

# def normalize_text(text):
#     """
#     Cleans text: lowercase and removes non-alphanumeric characters.
#     """
#     if not text:
#         return ""
#     return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()

# def get_parsed_dates(ner_data):
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
#     Returns a set of normalized source entities for fast O(1) exact matching.
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

# def check_fuzzy_match(summary_text_norm, source_entities_norm, threshold=85):
#     """
#     Compares a normalized summary entity against all normalized source entities.
#     Returns True if the highest fuzz ratio >= threshold.
#     """
#     for src_ent in source_entities_norm:
#         # fuzz.ratio returns an integer 0-100
#         if fuzz.ratio(summary_text_norm, src_ent) >= threshold:
#             return True
#     return False

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

# # ==========================================
# # 2. LHI CALCULATION (Date -> Exact -> Fuzzy)
# # ==========================================

# def calculate_lhi_details(source_ner, summary_ner):
#     """
#     LHI Calculation logic:
#     1. Date Parsing (for DATE labels)
#     2. Exact Match (Normalized)
#     3. Fuzzy Match (Ratio > 85)
#     """
#     # Pre-process source data
#     E_src_norm = get_flat_entity_set(source_ner) 
#     src_dt_objs = get_parsed_dates(source_ner)   

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
        
#         # A. Handle Dates
#         if label == "DATE":
#             if is_date_match(raw_text, src_dt_objs):
#                 is_match = True
        
#         # B. Non-Date or Date-Match-Failed: Try Exact Matching
#         if not is_match:
#             norm_text = normalize_text(raw_text)
#             if norm_text in E_src_norm:
#                 is_match = True
                
#             # C. Try Fuzzy Matching (Only if Exact Match failed)
#             elif check_fuzzy_match(norm_text, E_src_norm, threshold=85):
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
# OUTPUT_CSV = 'lhi_results_fuzzy.csv'

# try:
#     with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
#         source_data = json.load(f)
#     with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
#         summary_data = json.load(f)

#     print(f"Processing {len(summary_data)} documents using Fuzzy Logic...")
#     all_scores = []

#     with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(['Doc_ID', 'LHI_Score', 'Total_Summary_Entities', 'Hallucination_Count', 'Hallucinated_Entities'])
        
#         for doc_id, summary_ner in tqdm(summary_data.items(), desc="Processing"):
#             source_ner = source_data.get(doc_id, {})
#             score, total, h_count, h_list = calculate_lhi_details(source_ner, summary_ner)
            
#             if score != "N/A":
#                 all_scores.append(score)
            
#             h_list_str = "; ".join(h_list)
#             csv_writer.writerow([doc_id, score, total, h_count, h_list_str])

#     if all_scores:
#         avg_lhi = sum(all_scores) / len(all_scores)
#         print(f"\nGlobal Average LHI Score: {avg_lhi:.4f}")

# except Exception as e:
#     print(f"Error: {e}")




import json
import csv
import re
import dateparser
from tqdm import tqdm

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def normalize_text(text):
    """
    Standardizes text for exact matching by lowercasing 
    and removing special characters.
    """
    if not text:
        return ""
    # Fixed the variable name from previous snippet (text instead of s)
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()

def get_parsed_dates(ner_data):
    """
    Extracts and parses all date strings from the source.
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
    Creates a set of all source entities (normalized) 
    for high-speed O(1) exact match lookups.
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

def is_date_match(summary_date_str, source_dt_objects):
    """
    Checks if a summary date exists in the source date objects.
    """
    try:
        summ_dt = dateparser.parse(summary_date_str)
        if summ_dt is None:
            return False
        for src_dt in source_dt_objects:
            # Comparing only the date part (ignoring time)
            if summ_dt.date() == src_dt.date():
                return True
    except:
        return False
    return False

# ==========================================
# 2. LHI CALCULATION (Date & Exact Only)
# ==========================================

def calculate_lhi_details(source_ner, summary_ner):
    """
    LHI Calculation logic:
    1. If entity is a DATE: Use dateparser comparison.
    2. Otherwise: Use Exact String Matching (after normalization).
    """
    # Prepare source data for comparison
    E_src_norm = get_flat_entity_set(source_ner) 
    src_dt_objs = get_parsed_dates(source_ner)   

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
        
        # A. Check for Date Match
        if label == "DATE":
            if is_date_match(raw_text, src_dt_objs):
                is_match = True
        
        # B. Check for Exact Match (Normalized)
        # Note: We run this for non-dates OR if date parsing failed
        if not is_match:
            norm_text = normalize_text(raw_text)
            if norm_text in E_src_norm:
                is_match = True
        
        # If neither check passes, it is a hallucination
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
OUTPUT_CSV = 'lhi_results_exact_only.csv'

try:
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)

    print(f"Loaded {len(summary_data)} documents. Running Exact Match LHI...")
    all_scores = []

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Doc_ID', 'LHI_Score', 'Total_Summary_Entities', 'Hallucination_Count', 'Hallucinated_Entities'])
        
        for doc_id, summary_ner in tqdm(summary_data.items(), desc="Processing"):
            source_ner = source_data.get(doc_id, {})
            score, total, h_count, h_list = calculate_lhi_details(source_ner, summary_ner)
            
            if score != "N/A":
                all_scores.append(score)
            
            h_list_str = "; ".join(h_list)
            csv_writer.writerow([doc_id, score, total, h_count, h_list_str])

    if all_scores:
        avg_lhi = sum(all_scores) / len(all_scores)
        print(f"\nProcessing Complete!")
        print(f"Global Average LHI Score: {avg_lhi:.4f}")

except FileNotFoundError as e:
    print(f"Error: Could not find data files. {e}")
except Exception as e:
    print(f"An error occurred: {e}")