import torch
import json
import csv
import re
import dateparser
import os
from tqdm import tqdm
from thefuzz import fuzz
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import transformers

transformers.utils.logging.set_verbosity_error()

# ==========================================
# 1. LLM SETUP (Llama-3-8B)
# ==========================================
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" 
)

# Persistent Cache Logic
CACHE_FILE = "llm_equivalence_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        try:
            raw_cache = json.load(f)
            equivalence_cache = {tuple(eval(k)): v for k, v in raw_cache.items()}
        except: equivalence_cache = {}
else:
    equivalence_cache = {}

def save_cache():
    serializable_cache = {str(list(k)): v for k, v in equivalence_cache.items()}
    with open(CACHE_FILE, 'w') as f:
        json.dump(serializable_cache, f)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def normalize_text(text):
    if not text: return ""
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()

def get_parsed_dates(ner_data):
    dt_list = []
    if not ner_data or "DATE" not in ner_data: return dt_list
    for date_str in ner_data["DATE"]:
        try:
            dt = dateparser.parse(date_str)
            if dt: dt_list.append(dt)
        except: continue
    return dt_list

def is_date_match(summary_date_str, source_dt_objects):
    try:
        summ_dt = dateparser.parse(summary_date_str)
        if summ_dt is None: return False
        return any(summ_dt.date() == src_dt.date() for src_dt in source_dt_objects)
    except: return False

def ask_llm_equivalence(s1, s2):
    """
    Evaluates relationship using the original strict 7-parameter logic.
    """
    pair = tuple(sorted([str(s1), str(s2)]))
    if pair in equivalence_cache:
        return equivalence_cache[pair]

    system_message = (
        "You are a high-precision linguistic and legal analysis tool. "
        "Evaluate if two strings refer to the same specific entity, date, or case reference. "
        "Output strictly in JSON format with boolean values. No conversational filler."
    )

    user_query = f"""
Analyze String 1 and String 2 based on the following seven criteria. You must return a strict JSON object.

### CRITERIA DEFINITIONS:
1. entity_type_match:
   - TRUE if both strings represent the same class (e.g., both are Names, both are Case Citations, both are Dates).
   - FALSE if one is a name and the other is a location, etc.

2. numeric_identity:
   - If NO numbers are present, default to TRUE.
   - If numbers ARE present, TRUE only if the exact digits match (e.g., "Case 12" and "No. 12" is TRUE; "Case 12" and "Case 14" is FALSE).

3. numeric_role_consistency:
   - If NO numbers are present, default to TRUE.
   - TRUE if numbers in both strings serve the same purpose (e.g., both are years, or both are section numbers). 
   - FALSE if "1995" is a Year in String 1 but a Case Number in String 2.

4. naming_overlap:
   - TRUE if there is a shared unique identifier or "core" name (e.g., "Ram Pal Singh" and "Ram Singh" share the core "Ram Singh").
   - FALSE if the core identities differ (e.g., "John Doe" and "Jane Doe").

5. honorific_or_filler_variation:
   - TRUE if the ONLY differences are titles (Mr, Shri, Ji, Hon'ble) or legal boilerplate (No., of, at, the, vs).
   - FALSE if there are substantive differences in the names or numbers themselves.

6. semantic_equivalence:
   - The "Final Verdict": TRUE if a legal professional would treat these two strings as pointing to the exact same person, file, or date.
   - Example: "November 5, 1980" and "5/11/1980" are semantically equivalent.

7. contradiction_present:
   - TRUE if there is a factual conflict (e.g., different middle names, different years, different case numbers).
   - FALSE if they are simply variations of the same information.

### OUTPUT FORMAT:
Output strictly in JSON. No preamble. No explanation.
Example:
{{
  "entity_type_match": true,
  "numeric_identity": true,
  "numeric_role_consistency": true,
  "naming_overlap": true,
  "honorific_or_filler_variation": true,
  "semantic_equivalence": true,
  "contradiction_present": false
}}

Now evaluate:
String 1: "{s1}"
String 2: "{s2}"
Result:
"""
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_query}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=550, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not match: return False
        res = json.loads(match.group())
        
        is_valid_match = (
            res.get("entity_type_match") is True and
            res.get("numeric_identity") is True and
            res.get("numeric_role_consistency") is True and
            res.get("naming_overlap") is True and
            res.get("honorific_or_filler_variation") is True and
            res.get("semantic_equivalence") is True and
            res.get("contradiction_present") is False
        )
        equivalence_cache[pair] = is_valid_match
        return is_valid_match
    except:
        return False

# ==========================================
# 3. CORE LOGIC (Precision, Recall, F1)
# ==========================================

def calculate_metrics(source_ner, summary_ner):
    source_entities_list = []
    source_norm_set = set()
    for label, ents in source_ner.items():
        for e in ents:
            source_entities_list.append((e, label))
            source_norm_set.add(normalize_text(e))
    
    src_dt_objs = get_parsed_dates(source_ner)
    
    summary_entities_flat = []
    if summary_ner:
        for label, entities in summary_ner.items():
            for ent in entities:
                summary_entities_flat.append((ent, label))

    if not summary_entities_flat:
        return 0, 0, 0, 0, 0, []

    hit_source_indices = set()
    mismatched_with_types = []
    correct_summary_count = 0

    for summ_text, summ_label in summary_entities_flat:
        is_match = False
        norm_summ = normalize_text(summ_text)

        # 1. Date Check
        if summ_label == "DATE" and is_date_match(summ_text, src_dt_objs):
            is_match = True
        
        # 2. Exact/Fuzzy/LLM Check
        if not is_match:
            candidates = sorted(enumerate(source_entities_list), 
                                key=lambda x: fuzz.ratio(norm_summ, normalize_text(x[1][0])), reverse=True)
            
            for s_idx, (src_text, src_label) in candidates:
                # Early exit for efficiency
                if fuzz.ratio(norm_summ, normalize_text(src_text)) < 30: break
                
                # Check fuzzy or original LLM logic
                if fuzz.ratio(norm_summ, normalize_text(src_text)) > 85 or ask_llm_equivalence(summ_text, src_text):
                    is_match = True
                    hit_source_indices.add(s_idx)
                    break

        if is_match:
            correct_summary_count += 1
        else:
            mismatched_with_types.append(f"{summ_text} ({summ_label})")

    # Math
    precision = correct_summary_count / len(summary_entities_flat) if summary_entities_flat else 0
    recall = len(hit_source_indices) / len(source_entities_list) if source_entities_list else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, len(summary_entities_flat), len(mismatched_with_types), mismatched_with_types

# ==========================================
# 4. EXECUTION
# ==========================================

SOURCE_FILE = 'row_wise_legal_entities.json'
SUMMARY_FILE = 'row_wise_legal_entities_summ.json'
OUTPUT_CSV = 'lhi_f1_detailed_results.csv'

# 1. Load the full datasets
with open(SOURCE_FILE, 'r') as f: 
    source_data = json.load(f)
with open(SUMMARY_FILE, 'r') as f: 
    summary_data = json.load(f)

# 2. CORRECT SUBSETTING: Get the first 10 keys from the summary dictionary
subset_keys = list(summary_data.keys())[:3]

all_stats = []

with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Added Recall and F1 to headers
    writer.writerow(['Doc_ID', 'Precision_LHI', 'Recall', 'F1_Score', 'Total_Entities', 'Hallucination_Count', 'Mismatched_Entities_Types'])
    
    # 3. Iterate only through the 10 selected keys
    for doc_id in tqdm(subset_keys, desc="Evaluating 10 Documents"):
        # Get NER data for this specific document
        summary_ner = summary_data[doc_id]
        source_ner = source_data.get(doc_id, {}) # Lookup in source_data using the ID
        
        # Calculate P, R, F1 and get the list of mismatched entities with types
        p, r, f1, total, h_cnt, h_details = calculate_metrics(source_ner, summary_ner)
        
        # Write results to CSV
        writer.writerow([doc_id, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", total, h_cnt, " | ".join(h_details)])
        all_stats.append((p, r, f1))
        
        # Save cache every few documents to prevent LLM progress loss
        if len(all_stats) % 5 == 0: 
            save_cache()

# Final cache save
save_cache()

# 4. Print Average Results for the subset
if all_stats:
    avg_p = sum(s[0] for s in all_stats)/len(all_stats)
    avg_r = sum(s[1] for s in all_stats)/len(all_stats)
    avg_f1 = sum(s[2] for s in all_stats)/len(all_stats)
    print(f"\n--- Evaluation Complete (Subset: 10 Docs) ---")
    print(f"Average Precision (LHI): {avg_p:.4f}")
    print(f"Average Recall:         {avg_r:.4f}")
    print(f"Average F1-Score:       {avg_f1:.4f}")