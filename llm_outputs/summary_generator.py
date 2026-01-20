import os
import pandas as pd
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# 1. Configuration
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
INPUT_CSV = "legal_documents.csv"
OUTPUT_CSV = "summarized_legal_docs.csv"

# 2. Initialize Model
# Using 1.5-Flash for cost-efficiency when processing multiple documents
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)


def parse_document(doc_str):
    try:
        return ast.literal_eval(doc_str)
    except:
        return []

# 3. Define the Prompt
prompt_template = """
Write a professional legal summary of the following document. 
Highlight the parties, the core dispute, and the legal reasoning.

Document:
"{text}"

SUMMARY:"""
LEGAL_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

# 4. Load Summarization Chain
# 'stuff' is fastest for documents that fit in the 1M token window. 
# Use 'map_reduce' if your CSV rows contain extremely long text.
chain = load_summarize_chain(llm, chain_type="stuff", prompt=LEGAL_PROMPT)

# 5. Process CSV
df = pd.read_csv(INPUT_CSV)
docs_as_lists = df['document'].apply(parse_document).tolist()

# 3. OpenNyAI Data object expects a list of strings (the full text of each doc)
# So we join the sentences back together with spaces
texts_to_process = [" ".join(doc) for doc in docs_as_lists if doc][:5]

results = []

print(f"Starting summarization for {len(df)} documents...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    doc_id = row['doc_id']
    text_content = row['legal_text']
    
    if pd.isna(text_content) or str(text_content).strip() == "":
        results.append({"doc_id": doc_id, "summary": "Empty Document"})
        continue
    
    # Wrap text in a LangChain Document object
    doc = [Document(page_content=text_content)]
    
    try:
        summary = chain.run(doc)
        results.append({
            "doc_id": doc_id,
            "summary": summary.strip()
        })
    except Exception as e:
        print(f"Error processing {doc_id}: {e}")
        results.append({"doc_id": doc_id, "summary": "Error during generation"})

# 6. Save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"Summaries saved to {OUTPUT_CSV}")