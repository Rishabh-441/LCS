import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# 1. Configure 4-bit loading to avoid memory/shape issues
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" 
)

# 2. Define the inputs
s1 = "ram pal singh"
s2 = "ram ji"

# 3. Construct the Few-Shot Prompt
system_message = (
    "You are a high-precision linguistic and legal analysis tool. "
    "Evaluate if two strings refer to the same specific entity, date, or case reference. "
    "Output strictly in JSON format with boolean values. No conversational filler."
)

user_query = f"""
Evaluate the relationship between String 1 and String 2 based on:

- entity_type_match: Do they belong to the same category (e.g., Person, Case Ref, Date)?
- numeric_identity: If numbers exist, are the specific digits and values identical (e.g., 16 == 16)?
- numeric_role_consistency: Do the numbers represent the same thing (e.g., both are Year, or both are Case No)?
- naming_overlap: If names exist, is there a core match (e.g., 'Ram Pal Singh' contains 'Ram')?
- honorific_or_filler_variation: Is the difference only due to titles ('Ji', 'Mr') or legal fillers ('no.', 'of', 'at')?
- semantic_equivalence: Despite phrasing, do they point to the exact same factual record?
- contradiction_present: Is there a factual conflict (different year, different middle name, different number)?

Example 1:
String 1: "Ram Pal Singh"
String 2: "Ram Ji"
Result:
{{
  "entity_type_match": true,
  "numeric_identity": true,
  "numeric_role_consistency": true,
  "naming_overlap": true,
  "honorific_or_filler_variation": false,
  "semantic_equivalence": false,
  "contradiction_present": true
}}

Example 2:
String 1: "Appeal 16, 1995"
String 2: "Appeal no. 16 of 1995"
Result:
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

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_query},
]

# 4. Process and Generate
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# We use a low temperature for strict factual consistency
outputs = model.generate(
    **inputs, 
    max_new_tokens=100, 
    temperature=0.1,
    do_sample=False
)

# 5. Decode Output
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(f"Result: {response.strip()}")