import os
import time
import base64
import json
import pandas as pd
import requests
import datetime
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from dotenv import load_dotenv
# Import Google Generative AI for prompt translation
import google.generativeai as genai
# Load environment variables
load_dotenv()
# === CONFIG ===
API_KEY = os.environ.get('KRUTRIM_API_KEY')
CHITRA_ENDPOINT = "https://cloud.olakrutrim.com/v1/multimodal/generations/chitrarth"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Hyperparameters for chitrarth
MAX_TOKENS = 1024
REPETITION_PENALTY = 1.2
TOP_P = 0.9
TEMPERATURE = 0.1

# Gemini API settings for translations
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-2.0-flash'

# Cache for storing translated prompts
TRANSLATION_CACHE = {}
CACHE_FILE = "prompt_translation_cache_chitrarth.json"

# Set up logging directory
LOG_DIR = "inference_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Checkpoints directory for JSON format
CHECKPOINT_DIR = "checkpoints_json"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

# Image folder location
IMAGE_FOLDER = "../All_Images_formatted_final_dataset"

# Map CSV columns to output prefixes
QUESTION_COLUMNS = {
    "Refactored question":             "r",
    "Common Sense Cultural Question":  "cscq",
    "Multi-hop reasoning Question":    "mhr",
    "Analogy Question":                "an",
}

# Configure logger
def setup_logger(language, prefix):
    """Set up a specific logger for each language and question type combination"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{language}_{prefix}_chitrarth_{timestamp}.log")
    
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
1. {options[0]}
2. {options[1]}
3. {options[2]}
4. {options[3]}

Output Format:
After evaluating all cultural perspectives, respond with only the correct option (e.g. "1" or "Option1") and do not include any reasoning steps, chain-of-thought, or additional explanation.
""".strip()

# === Function to save and load checkpoints as JSON ===
def save_checkpoint_json(data, filename):
    """Save checkpoint data as JSON instead of pickle"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    try:
        # Convert DataFrame to dictionary
        if isinstance(data, pd.DataFrame):
            # Convert any NaN values to None for JSON serialization
            data_dict = data.where(pd.notnull(data), None).to_dict(orient='records')
            json_data = {"dataframe": data_dict}
        else:
            json_data = data
            
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"Saved checkpoint to {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Error saving checkpoint to JSON: {e}")
        return False

def load_checkpoint_json(filename):
    """Load checkpoint data from JSON"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            # If this is a DataFrame checkpoint, convert back to DataFrame
            if isinstance(json_data, dict) and "dataframe" in json_data:
                df = pd.DataFrame(json_data["dataframe"])
                return df
            return json_data
        except Exception as e:
            print(f"Error loading checkpoint from JSON: {e}")
            return None
    return None

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

def encode_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] encoding image {image_path}: {e}")
        return None

# Standard inference using Chitrarth API with translated prompt
def infer_standard(question: str, options: List[str], image_path: str, language: str = "English", logger=None) -> str:
    # Create standard prompt
    prompt_text = (
        f"Question: {question}\n"
        "Options:\n"
        f"1. {options[0]}\n"
        f"2. {options[1]}\n"
        f"3. {options[2]}\n"
        f"4. {options[3]}\n\n"
        "Based on the image and question, reply with only the correct option number (1–4) and no other text."
    )
    
    # Translate the prompt template to the target language
    if language != "English":
        prompt_text = translate_prompt(prompt_text, language)
    
    if logger:
        logger.info(f"{'='*50}")
        logger.info(f"IMAGE PATH: {image_path}")
        logger.info(f"{'='*50}")
        logger.info(f"PROMPT:")
        logger.info(f"{prompt_text}")
        logger.info(f"{'='*50}")
    
    # Encode image
    image_b64 = encode_image(image_path)
    if not image_b64:
        error_msg = "Error: failed to encode image"
        if logger:
            logger.error(error_msg)
        return error_msg
    
    # Prepare API request
    payload = {
        "modelName": "chitrarth",
        "prompt": prompt_text,
        "imageBase64": image_b64,
        "maxTokens": MAX_TOKENS,
        "repetitionPenalty": REPETITION_PENALTY,
        "top_p": TOP_P,
        "temperature": TEMPERATURE
    }
    
    # Debug payload without full base64
    debug_payload = payload.copy()
    debug_payload["imageBase64"] = f"[{len(image_b64)} chars]"
    if logger:
        logger.debug(f"Request payload: {json.dumps(debug_payload)}")
    
    try:
        r = requests.post(CHITRA_ENDPOINT, headers=HEADERS, json=payload)
        r.raise_for_status()
        
        if logger:
            logger.debug(f"Response status: {r.status_code}")
        
        data = r.json()
        # Handle generated_texts field
        if "generated_texts" in data:
            answer = data["generated_texts"].strip()
        # Handle generations field
        elif "generations" in data:
            gens = data["generations"]
            if isinstance(gens, list) and gens:
                first = gens[0]
                if isinstance(first, list) and first and isinstance(first[0], dict):
                    answer = first[0].get("text", "").strip()
                elif isinstance(first, dict):
                    answer = first.get("text", "").strip()
                else:
                    answer = f"Error: unexpected generations format {type(first)}"
            else:
                answer = f"Error: empty generations"
        # Fallback common fields
        else:
            for key in ("text", "result", "completion"):
                if key in data:
                    answer = str(data[key]).strip()
                    break
            else:
                answer = f"Error: unexpected response structure {list(data.keys())}"
        
        if logger:
            logger.info(f"MODEL OUTPUT: {answer}")
            logger.info(f"{'='*50}\n")
        
        return answer
    except Exception as e:
        error_msg = f"API error: {str(e)}"
        if logger:
            logger.error(error_msg)
        return error_msg

# Cultural-dimension inference using Chitrarth API with translated prompt
def infer_cultural(question: str, options: List[str], image_path: str, language: str = "English", logger=None) -> str:
    # Format the cultural prompt with question and options
    prompt_text = CULTURAL_PROMPT.format(question=question, options=options)
    
    # Translate the prompt template to the target language
    if language != "English":
        prompt_text = translate_prompt(prompt_text, language)
    
    if logger:
        logger.info(f"{'='*50}")
        logger.info(f"IMAGE PATH: {image_path}")
        logger.info(f"{'='*50}")
        logger.info(f"PROMPT:")
        logger.info(f"{prompt_text}")
        logger.info(f"{'='*50}")
    
    # Encode image
    image_b64 = encode_image(image_path)
    if not image_b64:
        error_msg = "Error: failed to encode image"
        if logger:
            logger.error(error_msg)
        return error_msg
    
    # Prepare API request
    payload = {
        "modelName": "chitrarth",
        "prompt": prompt_text,
        "imageBase64": image_b64,
        "maxTokens": MAX_TOKENS,
        "repetitionPenalty": REPETITION_PENALTY,
        "top_p": TOP_P,
        "temperature": TEMPERATURE
    }
    
    # Debug payload without full base64
    debug_payload = payload.copy()
    debug_payload["imageBase64"] = f"[{len(image_b64)} chars]"
    if logger:
        logger.debug(f"Request payload: {json.dumps(debug_payload)}")
    
    try:
        r = requests.post(CHITRA_ENDPOINT, headers=HEADERS, json=payload)
        r.raise_for_status()
        
        if logger:
            logger.debug(f"Response status: {r.status_code}")
        
        data = r.json()
        # Handle generated_texts field
        if "generated_texts" in data:
            answer = data["generated_texts"].strip()
        # Handle generations field
        elif "generations" in data:
            gens = data["generations"]
            if isinstance(gens, list) and gens:
                first = gens[0]
                if isinstance(first, list) and first and isinstance(first[0], dict):
                    answer = first[0].get("text", "").strip()
                elif isinstance(first, dict):
                    answer = first.get("text", "").strip()
                else:
                    answer = f"Error: unexpected generations format {type(first)}"
            else:
                answer = f"Error: empty generations"
        # Fallback common fields
        else:
            for key in ("text", "result", "completion"):
                if key in data:
                    answer = str(data[key]).strip()
                    break
            else:
                answer = f"Error: unexpected response structure {list(data.keys())}"
        
        if logger:
            logger.info(f"MODEL OUTPUT: {answer}")
            logger.info(f"{'='*50}\n")
        
        return answer
    except Exception as e:
        error_msg = f"API error: {str(e)}"
        if logger:
            logger.error(error_msg)
        return error_msg

# General processing function with logging
def process_column(qcol: str, prefix: str, df: pd.DataFrame, infer_fn, language: str) -> None:
    output_csv = f"{language}_{prefix}_output_with_answers_chitrarth.csv"
    checkpoint_file = f"checkpoint_{language}_{prefix}_chitrarth.json"  # Using JSON instead of pkl
    df_out = df.copy()
    df_out["ChosenOption"] = ""
    start_idx = 0

    # Set up logger for this run
    logger = setup_logger(language, prefix)
    logger.info(f"Starting inference for {language} - {prefix} at {datetime.datetime.now()}")
    logger.info(f"Processing column: {qcol}")
    logger.info(f"{'='*80}\n")

    # load checkpoint if exists
    checkpoint_df = load_checkpoint_json(checkpoint_file)
    if checkpoint_df is not None:
        df_out = checkpoint_df
        pending = df_out["ChosenOption"].isnull() | (df_out["ChosenOption"] == "")
        if pending.any():
            start_idx = df_out[pending].index[0]
            logger.info(f"Resuming from checkpoint at index {start_idx}")
        else:
            logger.info("All entries processed in checkpoint. Starting from beginning.")

    # prepare list of indices to process
    indices = list(df_out.index[start_idx:])
    
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
                # Add delay to avoid rate limiting
                time.sleep(1)
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

        df_out.at[idx, "ChosenOption"] = ans

        # write this row out
        row_df = df_out.loc[[idx], [qcol] + [f"Option{i}" for i in range(1,5)] + ["Image Name","ChosenOption"]]
        first = (idx == indices[0])
        row_df.to_csv(output_csv, mode="w" if first else "a", header=first, index=False)

        # checkpoint every 5 items (more frequent for API-based model)
        if (indices.index(idx) + 1) % 5 == 0:
            save_checkpoint_json(df_out, checkpoint_file)
            logger.info(f"Saved checkpoint at index {idx}")

    # final checkpoint
    save_checkpoint_json(df_out, checkpoint_file)
    logger.info(f"Completed inference for {language} - {prefix} at {datetime.datetime.now()}")
    logger.info(f"{'='*80}")

# Function to combine all outputs into a single CSV file per language
def combine_outputs():
    # Dictionary to store combined dataframes for each language
    combined_dfs = {}
    
    # Collect all CSV files
    for lang in CSV_PATHS.keys():
        lang_dfs = []
        
        # Collect all prefixes for this language
        for prefix in list(QUESTION_COLUMNS.values()) + ["cc"]:
            csv_file = f"{lang}_{prefix}_output_with_answers_chitrarth.csv"
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
        combined_file = f"{lang}_combined_chitrarth_results.csv"
        df.to_csv(combined_file, index=False)
        print(f"Created combined file: {combined_file}")
    
    return combined_dfs

# Entry point
if __name__ == "__main__":
    print(f"Starting inference with Chitrarth API")
    
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

                # standard inference
                process_column(qcol, prefix, df, infer_standard, lang)
                # cultural only for 'r'
                if prefix == "r":
                    process_column(qcol, "cc", df, infer_cultural, lang)

            print(f"\nGenerated CSV files for {lang}:")
            for p in list(QUESTION_COLUMNS.values()) + ["cc"]:
                print(f"- {lang}_{p}_output_with_answers_chitrarth.csv")
                
        except Exception as e:
            print(f"Error processing {lang}: {type(e).__name__}: {e}")
            continue  # Continue with next language if one fails
    
    # Combine all outputs into one CSV file per language
    print("\nCombining all outputs into one CSV file per language...")
    combine_outputs()
    print("\nDone!")
