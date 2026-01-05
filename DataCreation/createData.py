import os
import pandas as pd
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import argparse
from transformers import pipeline
from tqdm import tqdm
import torch

def process_file(input_file, output_file, generator, max_new_tokens, checkpoint_interval=50):
    try:
        df = pd.read_csv(input_file)
        if 'summary' not in df.columns:
            print(f"❌ {input_file} missing 'summary' column")
            return
    except Exception as e:
        print(f"❌ Could not read {input_file}: {e}")
        return

    # Resume if partial file exists
    if os.path.exists(output_file):
        out_df = pd.read_csv(output_file)
        if "super_summary" in out_df.columns:
            df["super_summary"] = out_df["super_summary"]
        else:
            df["super_summary"] = [None] * len(df)
    else:
        df["super_summary"] = [None] * len(df)

    # Check if all rows are already processed
    if df["super_summary"].notna().all():
        print(f"✅ Skipping {input_file}, already fully processed.")
        return

    # Prompt template
    template = """
        You are a friendly legal explainer. 
        Summarize the following text in **very simple and plain language**, as if explaining to someone with **no legal knowledge**. 

        Write your explanation as **one short paragraph**. 
        Do **not** write questions, answers, or bullet points. Just explain the case clearly.

        Cover these four points in your paragraph:
        1. Who was involved in the dispute.
        2. What the disagreement was about.
        3. What the final decision or outcome was.
        4. The main reason for that outcome.

        Here is the text to summarize:
        "{text_input}"
    """

    prompt = PromptTemplate(input_variables=["text_input"], template=template)

    # Sequential processing with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(input_file)}"):
        if pd.notna(row["super_summary"]):  # Skip already processed rows
            continue

        formatted_prompt = prompt.format(text_input=row["summary"])
        try:
            result = generator(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            full_text = result[0]["generated_text"]

            # Remove prompt text if included
            if full_text.startswith(formatted_prompt):
                answer = full_text[len(formatted_prompt):].strip()
            else:
                answer = full_text.strip()

            df.at[idx, "super_summary"] = answer
        except Exception as e:
            print(f"❌ Error in {input_file}, row {idx}: {e}")
            df.at[idx, "super_summary"] = "Error generating summary"

        # Save checkpoint every N rows
        if (idx + 1) % checkpoint_interval == 0:
            df.to_csv(output_file, index=False)

    # Final save
    df.to_csv(output_file, index=False)
    print(f"✅ Done: {output_file}")

# This is the block that will run on all files
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_prefix", default="iltur_dataset_part")
    parser.add_argument("--num_files", type=int, default=7)
    parser.add_argument("--max_tokens", type=int, default=300) # Increased from 200
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    args = parser.parse_args()

    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Load generator once
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device=device,
        torch_dtype=torch.float16 if device != -1 else torch.float32
    )

    # Sequentially process each file
    for i in range(1, args.num_files + 1):
        input_file = f"{args.input_prefix}{i}.csv"
        output_file = f"{args.input_prefix}{i}_processed.csv"
        if os.path.exists(input_file):
            # Note: We're removing the df.head(10) limit inside the function
            process_file(input_file, output_file, generator, args.max_tokens, args.checkpoint_interval)
        else:
            print(f"⚠️ Skipping {input_file} (not found)")


# if __name__ == "__main__":
#     load_dotenv()

#     MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
#     device = 0 if torch.cuda.is_available() else -1
#     generator = pipeline(
#         "text-generation",
#         model=MODEL_NAME,
#         device=device,
#         torch_dtype=torch.float16 if device != -1 else torch.float32
#     )

#     # Test only the first 10 rows of the first file
#     input_file = "iltur_dataset_part2.csv"
#     output_file = "iltur_dataset_part2_test.csv"
#     process_file(input_file, output_file, generator, max_new_tokens=300, checkpoint_interval=5)



# if __name__ == "__main__":
#     load_dotenv()

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_prefix", default="iltur_dataset_part")
#     parser.add_argument("--num_files", type=int, default=7)
#     parser.add_argument("--max_tokens", type=int, default=200)
#     parser.add_argument("--checkpoint_interval", type=int, default=50)
#     args = parser.parse_args()

#     MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

#     # Load generator once
#     device = 0 if torch.cuda.is_available() else -1
#     generator = pipeline(
#         "text-generation",
#         model=MODEL_NAME,
#         device=device,
#         torch_dtype=torch.float16 if device != -1 else torch.float32
#     )

#     # Sequentially process each file
#     for i in range(1, args.num_files + 1):
#         input_file = f"{args.input_prefix}{i}.csv"
#         output_file = f"{args.input_prefix}{i}_processed.csv"
#         if os.path.exists(input_file):
#             process_file(input_file, output_file, generator, args.max_tokens, args.checkpoint_interval)
#         else:
#             print(f"⚠️ Skipping {input_file} (not found)")
