import os
import pandas as pd
import torch
import gc
from tqdm import tqdm
from typing import List, Dict
import json
import datetime
import logging
from transformers import pipeline
from PIL import Image
# Import Google Generative AI for prompt translation
import google.generativeai as genai

# Set PyTorch memory management options
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

# Memory tracking function
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated, "
              f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB reserved")


GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-2.0-flash'  # or other appropriate model

# Cache for storing translated prompts
TRANSLATION_CACHE = {}
CACHE_FILE = "prompt_translation_cache_llama_vision.json"

# Set up logging to file
LOG_DIR = "inference_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logger
def setup_logger(language, prefix):
    """Set up a specific logger for each language and question type combination"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{language}_{prefix}_llama_vision_{timestamp}.log")
    
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

# === CONFIG ===
MODEL_PATH   = "kadirnar/Llama-3.2-1B-Vision"
IMAGE_FOLDER = "../All_Images_formatted_final_dataset"

# Multiple CSV files to process - mapping language names to file paths
CSV_PATHS = {
    "English": "../Corrected_Questions_Final_Dataset_english.csv",
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

# Map CSV columns to output prefixes
QUESTION_COLUMNS = {
    "Refactored question":             "r",
    "Common Sense Cultural Question":  "cscq",
    "Multi-hop reasoning Question":    "mhr",
    "Analogy Question":                "an",
}

pipe = pipeline(
    task="image-to-text",
    model=MODEL_PATH,
    device=0,
    torch_dtype=torch.float16,
    use_fast=True,
)
print("Model loaded successfully")
print_gpu_memory()

# === Cultural prompt template ===
CULTURAL_PROMPT = """
You are an expert analyst deeply knowledgeable in Indian culture, traditions, and regional heritage. Carefully analyze the provided image and question. Reason methodically through each of the following culturally informed dimensions to identify the correct answer. Please output only the correct option/answer from the given options without any additional information or reasoning steps.

Dimension A – Drishti (Visual Insight)
• Carefully examine the image, identifying culturally significant visual elements such as attire, architecture, rituals, landscapes, or symbols.

Dimension B – Smriti (Cultural Memory)
• Recall relevant historical details, traditional knowledge, or well-known cultural practices from India related to this question.

Dimension C – Yukti (Logical Integration)
• Logically integrate your observations from Drishti and knowledge from Smriti. Use this integration to rule out options that are culturally or logically inconsistent.

Dimension D – Sthiti (Regional Contextualization)
• Consider regional and cultural contexts within India. Determine which provided option best aligns with the cultural and regional insights you've gained.

Now, thoughtfully reason step-by-step through these cultural dimensions:

Question: {question}
Options:
1. {option1}
2. {option2}
3. {option3}
4. {option4}

Output Format:
After evaluating all cultural perspectives, respond with only the correct option (e.g. "1" or "Option1") and do not include any reasoning steps, chain-of-thought, or additional explanation.
""".strip()

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
            if line.startswith("Question:"):
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

# Standard inference using pipeline and original prompt format with translation
def infer_pipeline_standard(question: str, options: List[str], image_path: str, language: str = "English", logger=None) -> str:
    prompt = (
        f"Question: {question}\n"
        "Options:\n"
        f"1. {options[0]}\n"
        f"2. {options[1]}\n"
        f"3. {options[2]}\n"
        f"4. {options[3]}\n\n"
        "Based on the image and question, select and output the correct option. Do not output any other explanation."
    )
    
    # Translate the prompt template to the target language
    if language != "English":
        prompt = translate_prompt(prompt, language)
    
    if logger:
        logger.info(f"{'='*50}")
        logger.info(f"IMAGE PATH: {image_path}")
        logger.info(f"{'='*50}")
        logger.info(f"PROMPT:")
        logger.info(f"{prompt}")
        logger.info(f"{'='*50}")
    
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Run inference - for Llama-3.2-1B-Vision model
    output = pipe(images=image, prompt=prompt, max_new_tokens=16)
    answer = output[0].strip()
    
    if logger:
        logger.info(f"MODEL OUTPUT: {answer}")
        logger.info(f"{'='*50}\n")
    
    return answer

# Cultural-dimension inference using pipeline and translated prompt
def infer_pipeline_cultural(question: str, options: List[str], image_path: str, language: str = "English", logger=None) -> str:
    prompt = CULTURAL_PROMPT.format(question=question, option1=options[0], option2=options[1], option3=options[2], option4=options[3])
    
    # Translate the prompt template to the target language
    if language != "English":
        prompt = translate_prompt(prompt, language)
    
    if logger:
        logger.info(f"{'='*50}")
        logger.info(f"IMAGE PATH: {image_path}")
        logger.info(f"{'='*50}")
        logger.info(f"PROMPT:")
        logger.info(f"{prompt}")
        logger.info(f"{'='*50}")
    
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Run inference - for Llama-3.2-1B-Vision model
    output = pipe(images=image, prompt=prompt, max_new_tokens=16)
    answer = output[0].strip()
    
    if logger:
        logger.info(f"MODEL OUTPUT: {answer}")
        logger.info(f"{'='*50}\n")
    
    return answer

# General processing function with logging
def process_column(qcol: str, prefix: str, df: pd.DataFrame, infer_fn, language: str) -> None:
    output_csv = f"{language}_{prefix}_output_with_answers_llama_vision.csv"
    checkpoint  = f"checkpoint_{language}_{prefix}_llama_vision.pkl"
    df_out = df.copy()
    df_out["ChosenOption"] = ""
    start_idx = 0

    # Set up logger for this run
    logger = setup_logger(language, prefix)
    logger.info(f"Starting inference for {language} - {prefix} at {datetime.datetime.now()}")
    logger.info(f"Processing column: {qcol}")
    logger.info(f"{'='*80}\n")

    # load checkpoint if exists
    if os.path.exists(checkpoint):
        df_out = pd.read_pickle(checkpoint)
        pending = df_out["ChosenOption"].isnull() | (df_out["ChosenOption"] == "")
        if pending.any():
            start_idx = df_out[pending].index[0]
            logger.info(f"Resuming from checkpoint at index {start_idx}")
        else:
            logger.info("All entries processed in checkpoint. Starting from beginning.")

    # prepare list of indices to process
    indices = list(df_out.index[start_idx:])
    
    # Clear CUDA cache before processing
    torch.cuda.empty_cache()
    gc.collect()
    print_gpu_memory()
    
    # Only show progress bar in terminal, not in log
    for idx in tqdm(indices, desc=f"Inferring {language}_{prefix}", total=len(indices)):
        row = df_out.loc[idx]
        q = row[qcol]
        opts = [row[f"Option{i}"] for i in range(1,5)]
        img_path = os.path.join(IMAGE_FOLDER, row["Image Name"])
        
        # Log the question and options
        logger.info(f"{'='*50}")
        logger.info(f"INDEX: {idx}")
        logger.info(f"QUESTION: {q}")
        logger.info(f"OPTIONS:")
        for i, opt in enumerate(opts):
            logger.info(f"{i+1}. {opt}")
        logger.info(f"{'='*50}")

        if pd.isna(q) or not q.strip():
            ans = "Error: Missing Question"
            logger.info(f"ERROR: Missing question")
        elif not os.path.isfile(img_path):
            ans = f"Error: Missing Image '{row['Image Name']}'"
            logger.info(f"ERROR: Missing image - {row['Image Name']}")
        else:
            try:
                # Pass logger to inference function for prompt and response logging
                resp = infer_fn(q, opts, img_path, language, logger)
                # map numeric response to option text if needed
                if resp in {"1","2","3","4"}:
                    ans = opts[int(resp)-1]
                    logger.info(f"PARSED RESPONSE: Option {resp} = '{ans}'")
                else:
                    ans = resp
            except Exception as e:
                ans = f"Error: {type(e).__name__}: {e}"
                logger.info(f"EXCEPTION: {type(e).__name__}: {e}")
                logger.info(f"{'='*50}\n")
                
                # Try to recover from OOM errors
                if "CUDA out of memory" in str(e):
                    logger.info("CUDA OOM detected, cleaning memory...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    print_gpu_memory()

        df_out.at[idx, "ChosenOption"] = ans

        # write this row out
        row_df = df_out.loc[[idx], [qcol] + [f"Option{i}" for i in range(1,5)] + ["Image Name","ChosenOption"]]
        first = (idx == indices[0])
        row_df.to_csv(output_csv, mode="w" if first else "a", header=first, index=False)

        # checkpoint every 100 (reduced from 500 to save more frequently)
        if (indices.index(idx) + 1) % 100 == 0:
            df_out.to_pickle(checkpoint)
            logger.info(f"Saved checkpoint at index {idx}")
            
            # Clean memory periodically
            torch.cuda.empty_cache()
            gc.collect()
            print_gpu_memory()

    # final checkpoint
    df_out.to_pickle(checkpoint)
    logger.info(f"Completed inference for {language} - {prefix} at {datetime.datetime.now()}")
    logger.info(f"{'='*80}")
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    print_gpu_memory()

# Function to combine all outputs into a single CSV file per language
def combine_outputs():
    # Dictionary to store combined dataframes for each language
    combined_dfs = {}
    
    # Collect all CSV files
    for lang in CSV_PATHS.keys():
        lang_dfs = []
        
        # Collect all prefixes for this language
        for prefix in list(QUESTION_COLUMNS.values()) + ["cc"]:
            csv_file = f"{lang}_{prefix}_output_with_answers_llama_vision.csv"
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    # Add a column to indicate question type
                    df['QuestionType'] = prefix
                    lang_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
        
        # Combine all dataframes for this language
        if lang_dfs:
            combined_dfs[lang] = pd.concat(lang_dfs, ignore_index=True)
    
    # Save the combined dataframes
    for lang, df in combined_dfs.items():
        combined_file = f"{lang}_combined_llama_vision_results.csv"
        df.to_csv(combined_file, index=False)
        print(f"Created combined file: {combined_file}")
    
    return combined_dfs

# Entry point
if __name__ == "__main__":
    for lang, csv_path in CSV_PATHS.items():
        print(f"\nProcessing {lang} dataset from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            if not os.path.isdir(IMAGE_FOLDER):
                raise FileNotFoundError(f"Image folder '{IMAGE_FOLDER}' not found")

            for qcol, prefix in QUESTION_COLUMNS.items():
                required = [qcol] + [f"Option{i}" for i in range(1,5)] + ["Image Name"]
                if any(col not in df.columns for col in required):
                    print(f"Skipping '{prefix}' for '{lang}' (missing columns)")
                    continue
                    
                # Clean memory before starting a new column
                torch.cuda.empty_cache()
                gc.collect()
                print_gpu_memory()

                # standard inference
                process_column(qcol, prefix, df, infer_pipeline_standard, lang)
                # cultural only for 'r'
                if prefix == "r":
                    process_column(qcol, "cc", df, infer_pipeline_cultural, lang)

            print(f"\nGenerated CSV files for {lang}:")
            for p in list(QUESTION_COLUMNS.values()) + ["cc"]:
                print(f"- {lang}_{p}_output_with_answers_llama_vision.csv")
                
        except Exception as e:
            print(f"Error processing {lang}: {type(e).__name__}: {e}")
            continue  # Continue with next language if one fails
    
    # Combine all outputs into one CSV file per language
    print("\nCombining all outputs into one CSV file per language...")
    combine_outputs()
    print("\nDone!")
