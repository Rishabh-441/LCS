import pandas as pd
import ast
from opennyai import Pipeline
from opennyai.utils import Data
from rich.console import Console
from rich.table import Table
from rich import box
import json

# 1. Load the CSV
df = pd.read_csv('/home/rishabh/ThesisProject/DataCreation/merged_processed_output.csv')

# 2. Convert the "string of a list" into a real Python list
# Assuming your column is named 'document_text'
def parse_document(doc_str):
    try:
        return ast.literal_eval(doc_str)
    except:
        return []

def print_pretty_judgment(data):
    console = Console()
    
    # Create a table with specific styling
    table = Table(title="Legal Judgment Rhetorical Roles", 
                  box=box.ROUNDED, 
                  show_lines=True, 
                  title_style="bold magenta")

    table.add_column("Role", style="bold cyan", no_wrap=True, width=15)
    table.add_column("Sentence Text", style="white")

    # Mapping roles to specific colors for easier scanning
    role_colors = {
        "PREAMBLE": "grey70",
        "FAC": "green",
        "ANALYSIS": "yellow",
        "ARG_PETITIONER": "blue",
        "RATIO": "bold red",
        "RPC": "bold green",
        "NONE": "dim"
    }

    for entry in data:
        label = entry['labels'][0]
        text = entry['text']
        color = role_colors.get(label, "white")
        
        table.add_row(f"[{color}]{label}[/{color}]", text)

    console.print(table)


# Apply the conversion
docs_as_lists = df['document'].apply(parse_document).tolist()

# 3. OpenNyAI Data object expects a list of strings (the full text of each doc)
# So we join the sentences back together with spaces
texts_to_process = [" ".join(doc) for doc in docs_as_lists if doc][:5]

# 4. Run the Pipeline
if not texts_to_process:
    print("No documents found to process.")
else:
    # Set use_gpu=False if you don't have a dedicated NVIDIA GPU
    pipeline = Pipeline(components=['Rhetorical_Role'], use_gpu=True, verbose=True)
    data = Data(texts_to_process)
    results = pipeline(data)
    
    # Use it on your data
    data_list = results[0]['annotations']
    print_pretty_judgment(data_list)


    # 5. Extract and save predictions
    # complete_predictions = []

    # for i, result in enumerate(results):
    #     # This captures the full list of sentence-level dictionaries
    #     # containing: 'text', 'start', 'end', 'labels', 'id'
    #     full_annotation = result['annotations']
        
    #     complete_predictions.append({
    #         "doc_index": i,
    #         # We convert the list to a JSON string so it fits in a single CSV cell
    #         "full_prediction_json": json.dumps(full_annotation),
    #         "total_sentences": len(full_annotation)
    #     })

    # # 6. Save to a fresh, clean CSV
    # output_df = pd.DataFrame(complete_predictions)

    # save_path = '/home/rishabh/ThesisProject/Rhetorical Role Lableling/generated_data/complete_predictions.csv'
    # output_df.to_csv(save_path, index=False)

    # print(f"Done! Saved complete metadata for {len(complete_predictions)} docs.")
    # print(f"File location: {save_path}")