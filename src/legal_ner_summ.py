import spacy
import pandas as pd
import ast
import json
from collections import defaultdict
from tqdm import tqdm

# Load spaCy model
nlp = spacy.load("en_legal_ner_trf")

# Load CSV
df = pd.read_csv(
    "/home/rishabh/ThesisProject/DataCreation/merged_processed_output.csv"
)

# Final structure: row_no -> {label -> [entities]}
all_rows_entities = {}

# tqdm wrapped iterator
for row_idx, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Processing rows with Legal NER"
):
    try:
        # Parse document column (string -> list)
        texts = ast.literal_eval(row["summary"])
    except Exception:
        # Skip malformed rows safely
        continue

    entity_dict = defaultdict(set)

    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            entity_dict[ent.label_].add(ent.text.strip())

    # Convert sets → lists (JSON-safe)
    entity_dict_json = {
        label: sorted(list(ents))
        for label, ents in entity_dict.items()
    }

    all_rows_entities[row_idx] = entity_dict_json

# Save to JSON
output_path = "row_wise_legal_entities_summ.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_rows_entities, f, indent=2, ensure_ascii=False)

print(f"\nSaved entity dictionaries for {len(all_rows_entities)} rows")
print(f"Output file: {output_path}")
