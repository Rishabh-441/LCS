import torch
import json
import csv
import re
import dateparser
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

# Global Cache to prevent redundant LLM inferences across documents
equivalence_cache = {}

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
    Evaluates relationship using strict 7-parameter logic.
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
  "entity_type_match": bool,
  "numeric_identity": bool,
  "numeric_role_consistency": bool,
  "naming_overlap": bool,
  "honorific_or_filler_variation": bool,
  "semantic_equivalence": bool,
  "contradiction_present": bool
}}

Example 1:
String 1: "Ram Pal Singh" | String 2: "Ram Ji"
Result: {{"entity_type_match": true, "numeric_identity": true, "numeric_role_consistency": true, "naming_overlap": true, "honorific_or_filler_variation": false, "semantic_equivalence": false, "contradiction_present": true}}

Example 2:
String 1: "Appeal 16, 1995" | String 2: "Appeal no. 16 of 1995"
Result: {{"entity_type_match": true, "numeric_identity": true, "numeric_role_consistency": true, "naming_overlap": true, "honorific_or_filler_variation": true, "semantic_equivalence": true, "contradiction_present": false}}

Now evaluate:
String 1: "{s1}"
String 2: "{s2}"
Result:
"""
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_query}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        tokenizer.pad_token = tokenizer.eos_token
        outputs = model.generate(**inputs, max_new_tokens=550, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not match: return False
        res = json.loads(match.group())
        
        # Strict validation requirements as per prompt
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
        equivalence_cache[pair] = False
        return False

# ==========================================
# 3. CORE LOGIC (Tiered Hierarchy)
# ==========================================

def calculate_lhi_details(source_ner, summary_ner):
    source_entities_raw = []
    source_norm_set = set()
    for label, ents in source_ner.items():
        for e in ents:
            source_entities_raw.append(e)
            source_norm_set.add(normalize_text(e))
    
    src_dt_objs = get_parsed_dates(source_ner)

    summary_entities_flat = []
    if summary_ner:
        for label, entities in summary_ner.items():
            for ent in entities:
                summary_entities_flat.append((ent, label))

    if not summary_entities_flat: return "N/A", 0, 0, []

    hallucinated_entities = []
    
    for raw_text, label in summary_entities_flat:
        is_match = False
        norm_summary = normalize_text(raw_text)
        
        # STEP 1: Date Parsing
        if label == "DATE" and is_date_match(raw_text, src_dt_objs):
            is_match = True
        
        # STEP 2: Exact Match
        if not is_match:
            if norm_summary in source_norm_set:
                is_match = True
        
        # STEP 3: Fuzzy Matching (Ratio > 0.85)
        if not is_match:
            for src_ent in source_entities_raw:
                if fuzz.ratio(norm_summary, normalize_text(src_ent)) > 85:
                    is_match = True
                    break
        
        # STEP 4: LLM Strict Verification
        if not is_match:
            # Sort by fuzzy score to check most likely candidates first (Early Exit)
            candidates = sorted(source_entities_raw, key=lambda x: fuzz.ratio(raw_text, x), reverse=True)
            for src_ent in candidates:
                # If fuzzy is < 50, LLM is highly unlikely to find a strict 7-parameter match
                if fuzz.ratio(raw_text, src_ent) < 30: break 
                
                if ask_llm_equivalence(raw_text, src_ent):
                    is_match = True
                    break
        
        if not is_match:
            hallucinated_entities.append(raw_text)

    h_count = len(hallucinated_entities)
    total = len(summary_entities_flat)
    lhi_score = 1 - (h_count / total)
    return lhi_score, total, h_count, hallucinated_entities

# ==========================================
# 4. EXECUTION
# ==========================================

SOURCE_FILE = 'row_wise_legal_entities.json'
SUMMARY_FILE = 'row_wise_legal_entities_summ.json'
OUTPUT_CSV = 'lhi_final_llm.csv'

try:
    with open(SOURCE_FILE, 'r') as f: source_data = json.load(f)
    with open(SUMMARY_FILE, 'r') as f: summary_data = json.load(f)

    test_items = list(summary_data.items())
    all_scores = []

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Doc_ID', 'LHI_Score', 'Total_Entities', 'Hallucinations', 'Entity_List'])
        
        for doc_id, summary_ner in tqdm(test_items, desc="Evaluating Documents"):
            source_ner = source_data.get(doc_id, {})
            score, total, h_count, h_list = calculate_lhi_details(source_ner, summary_ner)
            
            if score != "N/A": all_scores.append(score)
            writer.writerow([doc_id, score, total, h_count, "; ".join(h_list)])
            print(f" Finished Doc {doc_id} | Score: {score}")

    if all_scores:
        print(f"\nSubset Test Average LHI: {sum(all_scores)/len(all_scores):.4f}")

except Exception as e:
    print(f"Error: {e}")