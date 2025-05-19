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
import json
# Import Google Generative AI for prompt translation
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-2.0-flash'  # or other appropriate model

# Cache for storing translated prompts
TRANSLATION_CACHE = {}
CACHE_FILE = "prompt_translation_cache.json"

# Load existing translation cache if available
def load_translation_cache():
    global TRANSLATION_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                TRANSLATION_CACHE = json.load(f)
            print(f"Loaded {len(TRANSLATION_CACHE)} cached translations")
        except Exception as e:
            print(f"Error loading translation cache: {e}")
            TRANSLATION_CACHE = {}
    else:
        TRANSLATION_CACHE = {}

# Save translation cache
def save_translation_cache():
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(TRANSLATION_CACHE, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(TRANSLATION_CACHE)} translations to cache")
    except Exception as e:
        print(f"Error saving translation cache: {e}")

# Setup Gemini API for translation
def setup_gemini():
    """Configure the Gemini API."""
    if not GEMINI_API_KEY:
        print("WARNING: No Gemini API key found. Set GEMINI_API_KEY in your .env file")
        return None
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    return model

# Initialize Gemini model
gemini_model = setup_gemini()

# Load translation cache at startup
load_translation_cache()


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
MODEL_PATH   = "moonshotai/Kimi-VL-A3B-Thinking"
IMAGE_FOLDER = "../All_Images_formatted_final_dataset"
# CSV_PATH     = "Corrected_Questions_Final_Dataset_English.csv"
CSV_PATHS = {
   # "English": "../Corrected_Questions_Final_Dataset_English.csv",
   # "Hindi": "../Corrected_Questions_Final_Dataset_Hindi.csv",
   # "Gujarati": "../Corrected_Questions_Final_Dataset_Gujarati.csv",
    "Malayalam": "../Corrected_Questions_Final_Dataset_Malayalam.csv",
   # "Tamil": "../Corrected_Questions_Final_Dataset_Tamil.csv",
   # "Telugu": "../Corrected_Questions_Final_Dataset_Telugu.csv",
   # "Bengali": "../Corrected_Questions_Final_Dataset_Bengali.csv",
   # "Marathi": "../Corrected_Questions_Final_Dataset_Marathi.csv",
   # "Punjabi": "../Corrected_Questions_Final_Dataset_Punjabi.csv",
   # "Odia": "../Corrected_Questions_Final_Dataset_Odia.csv",
   # "Assamese": "../Corrected_Questions_Final_Dataset_Assamese.csv",
   # "Urdu": "../Corrected_Questions_Final_Dataset_Urdu.csv",
   # "Kannada": "../Corrected_Questions_Final_Dataset_Kannada.csv",
   # "Konkani": "../Corrected_Questions_Final_Dataset_Konkani.csv",
   # "Sindhi": "../Corrected_Questions_Final_Dataset_Sindhi.csv",
}
BATCH_SIZE   = 2                  # ↑ tune for your GPU
GEN_CFG      = dict(max_new_tokens=128, do_sample=False, temperature=0.0)

QUESTION_COLUMNS = {
    # Skipping "Refactored question":            "r",
    # "Common Sense Cultural Question": "cscq",
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

# === Gemini Translation Function ===
def translate_prompt(text: str, target_language: str) -> str:
    """Translate the prompt template to the target language using Gemini API.
    
    Note: This will NOT translate the questions and options, as they are already
    translated in the dataset.
    """
    # Extract the template part from the full prompt
    if "{question}" in text:
        # For cultural prompt, this is a template that should be cached independently
        # of the specific question/options
        template = text
    else:
        # For standard prompt, extract just the template parts, not the specific question/options
        template_parts = []
        lines = text.split("\n")
        
        for line in lines:
            # Keep the template structure but remove specific content
            if line.startswith("<image_placeholder>"):
                template_parts.append(line)
            elif line.startswith("Question:"):
                template_parts.append("Question: {question}")
            elif line.startswith("Options:"):
                template_parts.append(line)
            elif any(line.startswith(f"{i}.") for i in range(1, 5)):
                # Replace option content with placeholder
                option_num = line.split(".")[0]
                template_parts.append(f"{option_num}. {{option{option_num}}}")
            elif "Based on the image and question" in line:
                template_parts.append(line)
            # Skip lines with specific question/option content
        
        template = "\n".join(template_parts)
    
    # Check cache for this template in this language
    cache_key = f"{target_language}:{template}"
    
    if cache_key in TRANSLATION_CACHE:
        translated_template = TRANSLATION_CACHE[cache_key]
        # Now reinsert the specific content into the translated template
        if "{question}" in text:
            # For cultural prompt with placeholders, just use as is
            return translated_template
        else:
            # For standard prompt, replace the placeholders with actual content
            result = translated_template
            # Find and replace question
            question_line = [line for line in text.split("\n") if line.startswith("Question:")][0]
            question = question_line.replace("Question: ", "")
            result = result.replace("{question}", question)
            
            # Find and replace options
            for i in range(1, 5):
                option_line = [line for line in text.split("\n") if line.startswith(f"{i}.")][0]
                option = option_line.replace(f"{i}. ", "")
                result = result.replace(f"{{option{i}}}", option)
                
            return result
    
    if gemini_model is None:
        print("Warning: Gemini model not available, skipping translation")
        return text
        
    try:
        # Extract parts that shouldn't be translated (placeholders and question/options)
        if "{question}" in text:
            # For templates with placeholders, we'll preserve them
            prompt = f"""
            Translate the following text to {target_language}. 
            DO NOT translate any text within curly braces like {{question}} or {{options[0]}}.
            Only translate the instructions and explanatory text:
            
            {text}
            """
        else:
            # For standard prompts, use template approach
            prompt = f"""
            Translate the following text to {target_language}.
            DO NOT translate the placeholders {{question}} or {{optionN}}.
            Only translate the instructions:
            
            {template}
            """
            
        response = gemini_model.generate_content(prompt)
        translated_template = response.text.strip()
        
        print(f"Translated prompt template to {target_language}")
        
        # Save the template translation to cache
        TRANSLATION_CACHE[cache_key] = translated_template
        save_translation_cache()
        
        # Now reinsert the specific content into the translated template
        if "{question}" in text:
            # For cultural prompt with placeholders, just use as is
            return translated_template
        else:
            # For standard prompt, replace the placeholders with actual content
            result = translated_template
            # Find and replace question
            question_line = [line for line in text.split("\n") if line.startswith("Question:")][0]
            question = question_line.replace("Question: ", "")
            result = result.replace("{question}", question)
            
            # Find and replace options
            for i in range(1, 5):
                option_line = [line for line in text.split("\n") if line.startswith(f"{i}.")][0]
                option = option_line.replace(f"{i}. ", "")
                result = result.replace(f"{{option{i}}}", option)
                
            return result
        
    except Exception as e:
        print(f"Translation error: {e}")
        # Return original text if translation fails
        return text

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
def process_column(qcol, prefix, df, language):
    out_csv = f"{language}_{prefix}_output_with_answers_kimivl.csv"
    ckpt    = f"checkpoint_{language}_{prefix}_kimivl.pkl"
    df_out  = df.copy()
    df_out["ChosenOption"] = ""
    start = 0
    if os.path.exists(ckpt):
        df_out = pd.read_pickle(ckpt)
        pending = df_out["ChosenOption"].eq("")
        if pending.any(): start = df_out[pending].index[0]

    all_indices = list(df_out.index[start:])
    bar = tqdm(range(0, len(all_indices), BATCH_SIZE),
               desc=f"KimiVL {prefix} {language}", unit="batch")

    for b_start in bar:
        b_ids   = all_indices[b_start:b_start+BATCH_SIZE]
        rows    = df_out.loc[b_ids]

        imgs, prompts, opt_lists, valid_ids = [], [], [], []

        for idx, row in rows.iterrows():
            q = row[qcol]
            opts = [row[f"Option{k}"] for k in range(1,5)]
            img_name = row["Image Name"]
            path = os.path.join(IMAGE_FOLDER, img_name) if str(img_name).strip().endswith(".jpg") else ""

            if not q or pd.isna(q):
                df_out.at[idx, "ChosenOption"] = "Error: Missing Question"
                continue
            if not path.startswith("http") and not os.path.isfile(path):
                df_out.at[idx, "ChosenOption"] = f"Error: Missing Image '{img_name}'"
                continue

            prompt = (CULTURAL_PROMPT.format(q=q, o1=opts[0], o2=opts[1], o3=opts[2], o4=opts[3])
            if prefix=="cc" else make_standard_prompt(q, opts))

            if language != "English":
                prompt = translate_prompt(prompt, language)

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
    for lang, csv_path in CSV_PATHS.items(): 
        # try:
            try:
                df = pd.read_csv(csv_path, encoding='ISO-8859-1')
            except UnicodeDecodeError:
                print("Failed to load with ISO-8859-1 encoding, trying windows-1252")
                df = pd.read_csv(csv_path, encoding='windows-1252')
            except FileNotFoundError:
                print(f"CSV file for {lang} not found: {csv_path}")
                continue

            if not os.path.isdir(IMAGE_FOLDER):
                raise FileNotFoundError(f"{IMAGE_FOLDER} not found")

            for qcol, prefix in QUESTION_COLUMNS.items():
                need = [qcol, *[f"Option{i}" for i in range(1,5)], "Image Name"]
                if any(c not in df.columns for c in need):
                    print(f"Skipping {prefix} for {lang} (missing columns)"); continue

                # Only process mhr and an question types
                process_column(qcol, prefix, df, lang)
                # Skip cultural context processing
                # if prefix == "r":                     # cultural run on refactored Qs
                #     process_column(qcol, "cc", df, lang)

            print("\nGenerated CSV files:")
            for p in list(QUESTION_COLUMNS.values()): # Removed "cc" since we're skipping it
                print(f"- {lang}_{p}_output_with_answers_kimivl.csv")
        # except Exception as e:
        #     print(f"Error processing {lang}: {type(e).__name__}: {e}")
        #     continue  # Continue with next language if one fails
