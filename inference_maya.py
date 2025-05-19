"""
Enhanced Maya inference script with support for:
- Multiple languages
- Logging functionality
- Prompt translation and caching
- Combined CSV output for all question types per language
"""

import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import json
import datetime
import logging
from PIL import Image     # required by talk-2-Maya loader
from huggingface_hub import login
# Import Google Generative AI for prompt translation
import google.generativeai as genai

# Replace with your actual Hugging Face API token
api_key = os.environ.get('HF_API_TOKEN')
login(token=api_key)

from llava.eval.talk2maya import run_vqa_model   # ← the new model API

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-2.0-flash'  # or other appropriate model

# Cache for storing translated prompts
TRANSLATION_CACHE = {}
CACHE_FILE = "prompt_translation_cache_maya.json"

# Set up logging to file
LOG_DIR = "inference_logs_maya"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logger
def setup_logger(language, prefix):
    """Set up a specific logger for each language and question type combination"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{language}_{prefix}_maya_{timestamp}.log")
    
    logger = logging.getLogger(f"{language}_{prefix}")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    
    return logger

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

# ---------------- user-specific paths ----------------
IMAGE_FOLDER = "../All_Images_formatted_final_dataset"

# Multiple CSV files to process - mapping language names to file paths
CSV_PATHS = {
    "Hindi": "../Corrected_Questions_Final_Dataset_Hindi.csv",
    "Gujarati": "../Corrected_Questions_Final_Dataset_Gujarati.csv",
    "Malayalam": "../Corrected_Questions_Final_Dataset_Malayalam.csv",
    "Tamil": "../Corrected_Questions_Final_Dataset_Tamil.csv",
    "Telugu": "../Corrected_Questions_Final_Dataset_Telugu.csv",
    "Bengali": "../Corrected_Questions_Final_Dataset_Bengali.csv",
    "Marathi": "../Corrected_Questions_Final_Dataset_Marathi.csv",
    "Punjabi": "../Corrected_Questions_Final_Dataset_Punjabi.csv",
    "Odia": "../Corrected_Questions_Final_Dataset_Odia.csv",
    "Assamese": "../Corrected_Questions_Final_Dataset_Assamese.csv",
    "Urdu": "../Corrected_Questions_Final_Dataset_Urdu.csv",
    "Kannada": "../Corrected_Questions_Final_Dataset_Kannada.csv",
    "Konkani": "../Corrected_Questions_Final_Dataset_Konkani.csv",
    "Sindhi": "../Corrected_Questions_Final_Dataset_Sindhi.csv",
}

QUESTION_COLUMNS = {
    "Refactored question":            "r",
    "Common Sense Cultural Question": "cscq",
    "Multi-hop reasoning Question":   "mhr",
    "Analogy Question":               "an",
}

# ---------------- Translation function ----------------
def translate_prompt(text: str, target_language: str) -> str:
    """Translate the prompt template to the target language using Gemini API.
    
    Note: This will NOT translate the questions and options, as they are already
    translated in the dataset.
    """
    # Extract the template part from the full prompt
    if any(placeholder in text for placeholder in ['{q}', '{o1}', '{o2}', '{o3}', '{o4}']):
        # For cultural prompt, this is a template that should be cached independently
        # of the specific question/options
        template = text
    else:
        # For standard prompts, extract just the template parts, not the specific question/options
        template_parts = []
        lines = text.split("\n")
        
        for line in lines:
            # Keep the template structure but remove specific content
            if line.startswith("Question:") or line.strip() == line and line and not line.startswith("Options:"):
                template_parts.append("{question}")
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
        if any(placeholder in text for placeholder in ['{q}', '{o1}', '{o2}', '{o3}', '{o4}']):
            # For cultural prompt with placeholders, just use as is
            return translated_template
        else:
            # For standard prompt, replace the placeholders with the original text
            original_parts = text.split("\n")
            translated_parts = translated_template.split("\n")
            result_parts = []
            
            for i, part in enumerate(translated_parts):
                if "{question}" in part:
                    # Find the corresponding original line with the question
                    for orig_line in original_parts:
                        if orig_line and not orig_line.startswith("Options:") and not any(orig_line.startswith(f"{j}.") for j in range(1, 5)):
                            result_parts.append(orig_line)
                            break
                elif "{option" in part:
                    # Extract option number and find the corresponding original option
                    opt_num = int(part.split("{")[1].split("}")[0].replace("option", ""))
                    for orig_line in original_parts:
                        if orig_line.startswith(f"{opt_num}."):
                            result_parts.append(orig_line)
                            break
                else:
                    result_parts.append(part)
            
            return "\n".join(result_parts)
    
    if gemini_model is None:
        print("Warning: Gemini model not available, skipping translation")
        return text
        
    try:
        # Extract parts that shouldn't be translated (placeholders and question/options)
        if any(placeholder in text for placeholder in ['{q}', '{o1}', '{o2}', '{o3}', '{o4}']):
            # For templates with placeholders, we'll preserve them
            prompt = f"""
            Translate the following text to {target_language}. 
            DO NOT translate any text within curly braces like {{q}} or {{o1}}.
            Only translate the instructions and explanatory text:
            
            {text}
            """
        else:
            # For standard prompts, use template approach
            prompt = f"""
            Translate the following text to {target_language}.
            DO NOT translate the placeholders {{question}} or {{optionN}}.
            Only translate the instructions, not the specific questions or options:
            
            {template}
            """
            
        response = gemini_model.generate_content(prompt)
        translated_template = response.text.strip()
        
        print(f"Translated prompt template to {target_language}")
        
        # Save the template translation to cache
        TRANSLATION_CACHE[cache_key] = translated_template
        save_translation_cache()
        
        # Now reinsert the specific content into the translated template
        if any(placeholder in text for placeholder in ['{q}', '{o1}', '{o2}', '{o3}', '{o4}']):
            # For cultural prompt with placeholders, just use as is
            return translated_template
        else:
            # For standard prompt, replace the placeholders with the original text
            original_parts = text.split("\n")
            translated_parts = translated_template.split("\n")
            result_parts = []
            
            for i, part in enumerate(translated_parts):
                if "{question}" in part:
                    # Find the corresponding original line with the question
                    for orig_line in original_parts:
                        if orig_line and not orig_line.startswith("Options:") and not any(orig_line.startswith(f"{j}.") for j in range(1, 5)):
                            result_parts.append(orig_line)
                            break
                elif "{option" in part:
                    # Extract option number and find the corresponding original option
                    opt_num = int(part.split("{")[1].split("}")[0].replace("option", ""))
                    for orig_line in original_parts:
                        if orig_line.startswith(f"{opt_num}."):
                            result_parts.append(orig_line)
                            break
                else:
                    result_parts.append(part)
            
            return "\n".join(result_parts)
        
    except Exception as e:
        print(f"Translation error: {e}")
        # Return original text if translation fails
        return text

# ---------------- prompts ----------------
def make_standard_prompt(q, opt):
    return (
        f"{q}\n\n"
        "Options:\n"
        f"1. {opt[0]}\n2. {opt[1]}\n3. {opt[2]}\n4. {opt[3]}\n\n"
        "Based on the image and question, select and output the correct option. "
        "Do not output any other explanation. ONLY OUTPUT THE CORRECT OPTION NUMBER."
    )

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
    "Question: {q}\n"
    "Options:\n1. {o1}\n2. {o2}\n3. {o3}\n4. {o4}\n\n"
    "After evaluating all perspectives respond with only the correct option (e.g. “1”). Do not output any reasoning steps, chain-of-thought, or additional explanation."
)

# ---------------- inference using talk-2-Maya with language support ----------------
def infer_standard(q: str, opts: List[str], img: str, language: str = "English", logger=None) -> str:
    prompt = make_standard_prompt(q, opts)
    
    # Translate the prompt template to the target language
    if language != "English":
        prompt = translate_prompt(prompt, language)
    
    if logger:
        logger.info(f"{'='*50}")
        logger.info(f"IMAGE PATH: {img}")
        logger.info(f"{'='*50}")
        logger.info(f"PROMPT:")
        logger.info(f"{prompt}")
        logger.info(f"{'='*50}")
    
    raw = run_vqa_model(question=prompt, image_file=img).strip()
    
    result = opts[int(raw[0])-1] if raw and raw[0] in "1234" else raw
    
    if logger:
        logger.info(f"MODEL OUTPUT: {raw}")
        logger.info(f"EXTRACTED ANSWER: {result}")
        logger.info(f"{'='*50}\n")
    
    return result

def infer_cultural(q: str, opts: List[str], img: str, language: str = "English", logger=None) -> str:
    prompt = CULTURAL_PROMPT.format(q=q, o1=opts[0], o2=opts[1], o3=opts[2], o4=opts[3])
    
    # Translate the prompt template to the target language
    if language != "English":
        prompt = translate_prompt(prompt, language)
    
    if logger:
        logger.info(f"{'='*50}")
        logger.info(f"IMAGE PATH: {img}")
        logger.info(f"{'='*50}")
        logger.info(f"PROMPT:")
        logger.info(f"{prompt}")
        logger.info(f"{'='*50}")
    
    raw = run_vqa_model(question=prompt, image_file=img).strip()
    
    result = opts[int(raw[0])-1] if raw and raw[0] in "1234" else raw
    
    if logger:
        logger.info(f"MODEL OUTPUT: {raw}")
        logger.info(f"EXTRACTED ANSWER: {result}")
        logger.info(f"{'='*50}\n")
    
    return result

# ---------------- Enhanced CSV processing with logging ----------------
def process_column(qcol: str, prefix: str, df: pd.DataFrame, infer_fn, language: str, results_dict: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
    out_csv = f"{language}_{prefix}_output_with_answers_maya.csv"
    ckpt    = f"checkpoint_{language}_{prefix}_maya.pkl"
    df_out  = df.copy()
    df_out["ChosenOption"] = ""
    start = 0
    
    # Set up logger for this run
    logger = setup_logger(language, prefix)
    logger.info(f"Starting inference for {language} - {prefix} at {datetime.datetime.now()}")
    logger.info(f"Processing column: {qcol}")
    logger.info(f"{'='*80}\n")
    
    if os.path.exists(ckpt):
        df_out = pd.read_pickle(ckpt)
        mask = df_out["ChosenOption"].eq("")
        if mask.any(): 
            start = df_out[mask].index[0]
            logger.info(f"Resuming from checkpoint at index {start}")
        else:
            logger.info("All entries processed in checkpoint. Starting from beginning.")

    idxs = list(df_out.index[start:])
    for idx in tqdm(idxs, total=len(idxs), desc=f"Maya {language}_{prefix}"):
        row   = df_out.loc[idx]
        q     = row[qcol]
        opts  = [row[f"Option{i}"] for i in range(1,5)]
        img   = row["Image Name"]
        path  = img if img.startswith("http") else os.path.join(IMAGE_FOLDER, img)
        
        # Log the question and options
        logger.info(f"{'='*50}")
        logger.info(f"INDEX: {idx}")
        logger.info(f"QUESTION: {q}")
        logger.info(f"OPTIONS:")
        for i, opt in enumerate(opts):
            logger.info(f"{i+1}. {opt}")
        logger.info(f"{'='*50}")

        if not q or pd.isna(q):
            ans = "Error: Missing Question"
            logger.info(f"ERROR: Missing question")
        elif not path.startswith("http") and not os.path.isfile(path):
            ans = f"Error: Missing Image '{img}'"
            logger.info(f"ERROR: Missing image - {img}")
        else:
            try: 
                ans = infer_fn(q, opts, path, language, logger)
            except Exception as e: 
                ans = f"Error: {type(e).__name__}: {e}"
                logger.info(f"EXCEPTION: {type(e).__name__}: {e}")
                logger.info(f"{'='*50}\n")

        df_out.at[idx, "ChosenOption"] = ans
        cols = [qcol, *[f"Option{i}" for i in range(1,5)], "Image Name", "ChosenOption"]
        df_out.loc[[idx], cols].to_csv(
            out_csv,
            mode="w" if idx == idxs[0] else "a",
            header = idx == idxs[0],
            index=False
        )
        
        # Checkpoint every 500 samples
        if (idxs.index(idx)+1) % 500 == 0:
            df_out.to_pickle(ckpt)
            logger.info(f"Saved checkpoint at index {idx}")
            
    # Final checkpoint
    df_out.to_pickle(ckpt)
    logger.info(f"Completed inference for {language} - {prefix} at {datetime.datetime.now()}")
    logger.info(f"{'='*80}")
    
    # Add to results dict if provided
    if results_dict is not None:
        results_dict[prefix] = df_out
    
    return df_out

# ---------------- main with multi-language support ----------------
if __name__ == "__main__":
    # Process each language CSV file
    for lang, csv_path in CSV_PATHS.items():
        print(f"\n{'-'*30}\nProcessing {lang} dataset...\n{'-'*30}")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"CSV file for {lang} not found: {csv_path}")
            continue
        
        if not os.path.isdir(IMAGE_FOLDER):
            raise FileNotFoundError(f"{IMAGE_FOLDER} not found")
            
        # Dictionary to store results for all question types for this language
        lang_results = {}
        
        for qcol, prefix in QUESTION_COLUMNS.items():
            need = [qcol, *[f"Option{i}" for i in range(1,5)], "Image Name"]
            if any(c not in df.columns for c in need):
                print(f"Skipping '{prefix}' for '{lang}' (missing columns)")
                continue

            # Standard inference - store results in lang_results
            process_column(qcol, prefix, df, infer_standard, lang, lang_results)
            
            # Cultural only for 'r'
            if prefix == "r":
                process_column(qcol, "cc", df, infer_cultural, lang, lang_results)
        
        # Create a combined CSV file with all question types for this language
        print(f"\nCreating combined CSV file for {lang}...")
        combined_output = f"{lang}_combined_maya_results.csv"
        
        # All columns that will be in the combined CSV
        all_cols = []
        for qtype, df_result in lang_results.items():
            # Get questions columns that exist in this result
            q_cols = [col for col in QUESTION_COLUMNS.keys() if col in df_result.columns]
            for col in q_cols:
                all_cols.extend([col, "ChosenOption_" + qtype])
        all_cols = list(dict.fromkeys(all_cols))  # Remove duplicates while preserving order
        
        # Add option columns and image name
        option_cols = [f"Option{i}" for i in range(1,5)]
        all_cols.extend(option_cols + ["Image Name"])
        
        # Create empty combined DataFrame
        df_combined = pd.DataFrame(index=df.index)
        
        # Add image name and options that should be consistent across all question types
        df_combined["Image Name"] = df["Image Name"]
        for opt_col in option_cols:
            df_combined[opt_col] = df[opt_col]
        
        # Add each question type's data
        for qtype, df_result in lang_results.items():
            for qcol in QUESTION_COLUMNS.keys():
                if qcol in df_result.columns:
                    df_combined[qcol] = df_result[qcol]
                    df_combined[f"ChosenOption_{qtype}"] = df_result["ChosenOption"]
        
        # Save combined results
        df_combined.to_csv(combined_output, index=False)
        
        print(f"Generated combined CSV file for {lang}:")
        print(f"- {combined_output}")
        
        print(f"\nGenerated individual CSV files for {lang}:")
        for p in list(QUESTION_COLUMNS.values()) + ["cc"]:
            if p in lang_results or (p == "cc" and "r" in lang_results):
                print(f"- {lang}_{p}_output_with_answers_maya.csv")
    
    print("\nAll processing complete!")
