import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import List
from transformers import pipeline

# === CONFIG ===
MODEL_PATH   = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_FOLDER = "All_Images_formatted_final_dataset"
CSV_PATH     = "Corrected_Questions - FINAL_DATASET_ENGLISH.csv"

# Map CSV columns to output prefixes
QUESTION_COLUMNS = {
    "Refactored question":             "r",
    "Common Sense Cultural Question":  "cscq",
    "Multi-hop reasoning Question":    "mhr",
    "Analogy Question":                "an",
}

# === Load pipeline with 4-bit quantization ===
pipe = pipeline(
    task="image-text-to-text",
    model=MODEL_PATH,
    device=0,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)

# Standard inference using pipeline and original prompt format
def infer_pipeline_standard(question: str, options: List[str], image_path: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text",  "text": (
                    f"<image_placeholder>\n"
                    f"Question: {question}\n"
                    "Options:\n"
                    f"1. {options[0]}\n"
                    f"2. {options[1]}\n"
                    f"3. {options[2]}\n"
                    f"4. {options[3]}\n\n"
                    "Based on the image and question, select and output the correct option. Do not output any other explanation."
                )},
            ]
        }
    ]
    output = pipe(text=messages, max_new_tokens=16, return_full_text=False)
    return output[0]['generated_text'].strip()

# Cultural-dimension inference using pipeline and original prompt
def infer_pipeline_cultural(question: str, options: List[str], image_path: str) -> str:
    CULTURAL_PROMPT = (
        "You are an expert analyst deeply knowledgeable in Indian culture, traditions, and regional heritage. "
        "Carefully analyze the provided image and question. Reason methodically through each of the following culturally informed dimensions to identify the correct answer. "
        "Please output only the correct option/answer from the given options without any additional information or reasoning steps.\n\n"
        "Dimension A – Drishti (Visual Insight)\n"
        "•  Carefully examine the image, identifying culturally significant visual elements such as attire, architecture, rituals, landscapes, or symbols.\n\n"
        "Dimension B – Smriti (Cultural Memory)\n"
        "•  Recall relevant historical details, traditional knowledge, or well-known cultural practices from India related to this question.\n\n"
        "Dimension C – Yukti (Logical Integration)\n"
        "•  Logically integrate your observations from Drishti and knowledge from Smriti. Use this integration to rule out options that are culturally or logically inconsistent.\n\n"
        "Dimension D – Sthiti (Regional Contextualization)\n"
        "•  Consider regional and cultural contexts within India. Determine which provided option best aligns with the cultural and regional insights you've gained.\n\n"
        f"Question: {question}\n"
        f"Options:\n1. {options[0]}\n2. {options[1]}\n3. {options[2]}\n4. {options[3]}\n\n"
        "Output Format:\nAfter evaluating all cultural perspectives, respond with only the correct option (e.g. “1” or “Option1”) and do not include any reasoning steps, chain-of-thought, or additional explanation."
    )
    messages = [
        {"role": "user", "content": [
            {"type": "image", "url": image_path},
            {"type": "text",  "text": CULTURAL_PROMPT}
        ]}
    ]
    output = pipe(text=messages, max_new_tokens=16, return_full_text=False)
    return output[0]['generated_text'].strip()

# General processing function
def process_column(qcol: str, prefix: str, df: pd.DataFrame, infer_fn) -> None:
    output_csv = f"{prefix}_output_with_answers_qwenvl.csv"
    checkpoint  = f"checkpoint_{prefix}_qwenvl.pkl"
    df_out = df.copy()
    df_out["ChosenOption"] = ""
    start_idx = 0
    if os.path.exists(checkpoint):
        df_out = pd.read_pickle(checkpoint)
        pending = df_out['ChosenOption'].isnull() | (df_out['ChosenOption'] == '')
        if pending.any(): start_idx = df_out[pending].index[0]

    for idx in tqdm(df_out.index[start_idx:], desc=f"Inferring {prefix}"):
        row = df_out.loc[idx]
        question = row[qcol]
        opts     = [row[f"Option{i}"] for i in range(1,5)]
        img_name = row["Image Name"]
        # determine full image URL or path
        full_img = img_name if img_name.startswith("http") else os.path.abspath(os.path.join(IMAGE_FOLDER, img_name))
        if pd.isna(question) or not question.strip():
            answer = "Error: Missing Question"
        elif not os.path.isfile(full_img) and not full_img.startswith("http"):
            answer = f"Error: Missing Image {img_name}"
        else:
            try:
                resp = infer_fn(question, opts, full_img)
                answer = opts[int(resp[0])-1] if resp and resp[0] in '1234' else resp
            except Exception as e:
                answer = f"Error: {type(e).__name__}: {e}"
        df_out.at[idx, "ChosenOption"] = answer
        row_df = df_out.loc[[idx], [qcol] + [f"Option{i}" for i in range(1,5)] + ["Image Name","ChosenOption"]]
        header = idx == df_out.index[start_idx]
        row_df.to_csv(output_csv, mode="w" if header else "a", header=header, index=False)
        if ((idx - df_out.index[0] + 1) % 500) == 0:
            df_out.to_pickle(checkpoint)
    df_out.to_pickle(checkpoint)

# Entry point
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(f"Image folder '{IMAGE_FOLDER}' not found")
    for qcol, prefix in QUESTION_COLUMNS.items():
        required = [qcol] + [f"Option{i}" for i in range(1,5)] + ["Image Name"]
        if any(col not in df.columns for col in required):
            print(f"Skipping '{prefix}' (missing columns)")
            continue
        process_column(qcol, prefix, df, infer_pipeline_standard)
        if prefix == 'r':
            process_column(qcol, 'cc', df, infer_pipeline_cultural)
    print("\nGenerated CSV files:")
    for pref in list(QUESTION_COLUMNS.values()) + ['cc']:
        print(f"- {pref}_output_with_answers.csv")
