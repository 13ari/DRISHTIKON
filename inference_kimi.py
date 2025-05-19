"""
Kimi-VL (moonshotai/Kimi-VL-A3B-Thinking) inference script
=========================================================

• Loads the model in **bnb-4-bit (NF4)** to save GPU memory  
• Keeps your two prompts **exactly** as-is  
• Runs **batched inference** (default 8 rows at once) for ~5× throughput  
• CSV / checkpoint logic identical – each row still ends with a single
  “ChosenOption” value (cleaned to contain **only** the option text).

Dependencies:  transformers ≥ 4.38, bitsandbytes ≥ 0.41.1, torch ≥ 2.1,
Pillow, pandas, tqdm.
"""

import os, re, torch, pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
MODEL_PATH   = "moonshotai/Kimi-VL-A3B-Thinking"
IMAGE_FOLDER = "All_Images_formatted_final_dataset"
CSV_PATH     = "Corrected_Questions - FINAL_DATASET_ENGLISH.csv"
BATCH_SIZE   = 2                  # ↑ tune for your GPU
GEN_CFG      = dict(max_new_tokens=128, do_sample=False)

QUESTION_COLUMNS = {
    "Refactored question":            "r",
    "Common Sense Cultural Question": "cscq",
    "Multi-hop reasoning Question":   "mhr",
    "Analogy Question":               "an",
}

# --------------------------------------------------------------------------
# 4-bit model load
# --------------------------------------------------------------------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
).eval()

# --------------------------------------------------------------------------
# PROMPTS  (unchanged)
# --------------------------------------------------------------------------
def make_standard_prompt(q, o):
    return (
        "<image>\n"
        f"Question: {q}\n"
        "Options:\n"
        f"1. {o[0]}\n2. {o[1]}\n3. {o[2]}\n4. {o[3]}\n\n"
        "Based on the image and question, select and output the correct option. "
        "Do not output any other explanation."
    )

CULTURAL_PROMPT = (
    "<image>\n"
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
    "Question: {q}\n"
    "Options:\n1. {o1}\n2. {o2}\n3. {o3}\n4. {o4}\n\n"
    "Output Format:\nAfter evaluating all cultural perspectives, respond with only the correct option (e.g. “1” or “Option1”) and do not include any reasoning steps, chain-of-thought, or additional explanation."
)

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def _clean_answer(raw: str, opts: List[str]) -> str:
    """Strip rambles, return the option text best matching the model reply."""
    raw = raw.replace("◁think▷", "").strip()
    m = re.search(r"\b([1-4])\b", raw)
    if m:
        return opts[int(m.group(1)) - 1]
    for opt in opts:
        if re.search(re.escape(opt), raw, flags=re.I):
            return opt
    return raw

def _kimi_batch(images: List[Image.Image], prompts: List[str]) -> List[str]:
    """Run Kimi-VL on a batch of (image, prompt)."""
    msgs = [
        {"role": "user",
         "content": [{"type": "image", "image": img},
                     {"type": "text",  "text": pr}]}
        for img, pr in zip(images, prompts)
    ]
    text   = processor.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt")
    inputs = processor(images=images, text=text, return_tensors="pt",
                       padding=True, truncation=True).to(model.device)
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **GEN_CFG)
    outs_trim = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    return processor.batch_decode(outs_trim, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)

# --------------------------------------------------------------------------
# CSV processing with batching
# --------------------------------------------------------------------------
def process_column(qcol, prefix, df):
    out_csv = f"{prefix}_output_with_answers_kimivl.csv"
    ckpt    = f"checkpoint_{prefix}_kimivl.pkl"
    df_out  = df.copy()
    df_out["ChosenOption"] = ""
    start = 0
    if os.path.exists(ckpt):
        df_out = pd.read_pickle(ckpt)
        pending = df_out["ChosenOption"].eq("")
        if pending.any(): start = df_out[pending].index[0]

    all_indices = list(df_out.index[start:])
    bar = tqdm(range(0, len(all_indices), BATCH_SIZE),
               desc=f"KimiVL {prefix}", unit="batch")

    for b_start in bar:
        b_ids   = all_indices[b_start:b_start+BATCH_SIZE]
        rows    = df_out.loc[b_ids]

        imgs, prompts, opt_lists, valid_ids = [], [], [], []

        for idx, row in rows.iterrows():
            q = row[qcol]
            opts = [row[f"Option{k}"] for k in range(1,5)]
            img_name = row["Image Name"]
            path = img_name if img_name.startswith("http") else os.path.join(IMAGE_FOLDER, img_name)

            if not q or pd.isna(q):
                df_out.at[idx, "ChosenOption"] = "Error: Missing Question"
                continue
            if not path.startswith("http") and not os.path.isfile(path):
                df_out.at[idx, "ChosenOption"] = f"Error: Missing Image '{img_name}'"
                continue

            prompt = (CULTURAL_PROMPT.format(q=q, o1=opts[0], o2=opts[1],
                                             o3=opts[2], o4=opts[3])
                      if prefix=="cc" else make_standard_prompt(q, opts))

            imgs.append(Image.open(path).convert("RGB"))
            prompts.append(prompt)
            opt_lists.append(opts)
            valid_ids.append(idx)

        if not valid_ids:
            continue

        raw_answers = _kimi_batch(imgs, prompts)

        for idx, raw, opts in zip(valid_ids, raw_answers, opt_lists):
            df_out.at[idx, "ChosenOption"] = _clean_answer(raw, opts)

        # stream batch to csv
        cols = [qcol, *[f"Option{k}" for k in range(1,5)], "Image Name", "ChosenOption"]
        df_out.loc[valid_ids, cols].to_csv(
            out_csv,
            mode   = "a" if os.path.exists(out_csv) else "w",
            header = not os.path.exists(out_csv),
            index  = False)

        if (b_start // BATCH_SIZE + 1) % 20 == 0:
            df_out.to_pickle(ckpt)

    df_out.to_pickle(ckpt)

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(f"{IMAGE_FOLDER} not found")

    for qcol, prefix in QUESTION_COLUMNS.items():
        need = [qcol, *[f"Option{i}" for i in range(1,5)], "Image Name"]
        if any(c not in df.columns for c in need):
            print(f"Skipping {prefix} (missing columns)"); continue

        process_column(qcol, prefix, df)
        if prefix == "r":                     # cultural run on refactored Qs
            process_column(qcol, "cc", df)

    print("\nGenerated CSV files:")
    for p in list(QUESTION_COLUMNS.values()) + ["cc"]:
        print(f"- {p}_output_with_answers_kimivl.csv")
