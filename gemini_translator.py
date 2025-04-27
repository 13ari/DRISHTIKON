import csv
import os
import pandas as pd
import google.generativeai as genai
import time
from dotenv import load_dotenv
import random
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration variables - modify these as needed
INPUT_FILE = 'Generated_Questions.csv'                             # Input CSV file
OUTPUT_DIR = 'translated'                           # Directory to store translated files
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')   # API key from .env file
MODEL = 'gemini-2.5-flash-preview-04-17'                 # Gemini model to use
REPLACE_ORIGINAL = True                             # Replace original content instead of adding new columns

# Columns to translate (by index) - adjust these for your CSV
COLUMNS_TO_TRANSLATE = [3, 4, 5, 6, 7, 8,  12, 13, 14]                       # Indices of columns to translate (0-based index)

# Target languages for translation 
LANGUAGES = [
    'Hindi', 
    'Punjabi',
    'Urdu',
    'Telugu',
    'Tamil',
    'Kannada',
    'Malayalam',
    'Bengali',
    'Assamese',
    'Odia',
    'Marathi',
    'Gujarati',
    'Konkani',
    'Sindhi'
]

# Rate limiting settings
INITIAL_DELAY_BETWEEN_ROWS = 0.1        # Start with this many seconds between rows
MAX_DELAY = 10                        # Maximum delay in seconds
BACKOFF_FACTOR = 1.1                    # Multiply delay by this factor on error
JITTER = 0.1                          # Add random jitter (0-0.1 seconds) to avoid thundering herd

# Checkpoint settings
RESUME_FROM_CHECKPOINT = True         # Whether to resume from checkpoint
CHECKPOINT_DIR = "checkpoints"        # Directory to store checkpoints

def setup_gemini():
    """Configure the Gemini API."""
    if not GEMINI_API_KEY:
        raise ValueError("No Gemini API key found. Set GEMINI_API_KEY in your .env file")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL)
    return model

def translate_text(model, text, target_language, current_delay):
    """Translate text to target language using Gemini with exponential backoff."""
    if not text or not text.strip():
        return "", current_delay
    
    max_retries = 5
    retries = 0
    delay = current_delay
    
    while retries < max_retries:
        try:
            prompt = f"Translate the following text to {target_language}. Only return the translated text, no additional explanations or notes:\n\n{text}"
            response = model.generate_content(prompt)
            return response.text.strip(), current_delay  # Return current delay if successful
        
        except Exception as e:
            retries += 1
            error_msg = str(e)
            print(f"Error translating to {target_language} (attempt {retries}/{max_retries}): {error_msg}")
            
            if "429" in error_msg:  # Rate limit error
                # Increase delay for future requests
                delay = min(delay * BACKOFF_FACTOR, MAX_DELAY)
                wait_time = delay + random.uniform(0, JITTER)
                print(f"Rate limit hit. Increasing delay to {delay:.1f}s. Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
            else:
                # For other errors, wait a bit but don't increase delay
                print(f"Waiting {delay}s before retry...")
                time.sleep(delay)
    
    # If we got here, we failed after max retries
    return f"ERROR: Translation failed after {max_retries} attempts: {error_msg}", delay

def create_checkpoint(language, last_row_idx, current_delay):
    """Save checkpoint information to resume later."""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{os.path.splitext(os.path.basename(INPUT_FILE))[0]}_{language}_checkpoint.json")
    checkpoint_data = {
        "language": language,
        "last_row_processed": last_row_idx,
        "current_delay": current_delay,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    
    print(f"Saved checkpoint at row {last_row_idx} for {language}")

def load_checkpoint(language):
    """Load checkpoint information to resume processing."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{os.path.splitext(os.path.basename(INPUT_FILE))[0]}_{language}_checkpoint.json")
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        print(f"Resuming {language} from row {checkpoint_data['last_row_processed']+1} with delay {checkpoint_data['current_delay']}s")
        return checkpoint_data["last_row_processed"], checkpoint_data["current_delay"]
    
    return -1, INITIAL_DELAY_BETWEEN_ROWS  # No checkpoint exists

def create_output_file(input_file, language, resume_row=-1):
    """Create an output CSV file for a specific language."""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Generate output filename
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(OUTPUT_DIR, f"{base_filename}_{language}.csv")
    
    # If we're not resuming or the file doesn't exist, create a new file with header
    if resume_row == -1 or not os.path.exists(output_file):
        # Read input CSV to get column names
        df = pd.read_csv(input_file)
        header = list(df.columns)
        
        # Write header to output file (no additional columns needed)
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
        print(f"\nCreated new output file for {language}: {output_file}")
    else:
        print(f"\nUsing existing output file for {language}: {output_file}")
        
        # Read existing file to get headers
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
    
    return output_file, header

def process_csv_for_language(input_file, language, model):
    """Process the CSV for a specific language with advanced error handling and resumption."""
    # Check if we should resume from checkpoint
    start_row, current_delay = load_checkpoint(language) if RESUME_FROM_CHECKPOINT else (-1, INITIAL_DELAY_BETWEEN_ROWS)
    
    # Create output file with headers for this language
    output_file, header = create_output_file(input_file, language, start_row)
    
    # Read input data
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        all_rows = list(reader)
    
    total_rows = len(all_rows)
    print(f"Found {total_rows} rows to process for {language}")
    print(f"Starting with delay between requests: {current_delay}s")
    
    # Process each row
    for row_idx, row in enumerate(all_rows):
        # Skip already processed rows if resuming
        if row_idx <= start_row:
            continue
            
        # Create a new row starting with the original data
        new_row = row.copy()
        print(f"Processing row {row_idx+1}/{total_rows} for {language}")
        
        # Translate specified columns and replace the original content
        for col_idx in COLUMNS_TO_TRANSLATE:
            if col_idx < len(row) and row[col_idx]:
                original_text = row[col_idx]
                col_name = header[col_idx] if col_idx < len(header) else f"Column_{col_idx}"
                
                print(f"  Translating {col_name} to {language}...")
                translated, current_delay = translate_text(model, original_text, language, current_delay)
                
                # Replace original content with translated content
                new_row[col_idx] = translated
                
                # Print a preview of the translated text
                if len(translated) > 40:
                    print(f"  ✓ {language}: {translated[:40]}...")
                else:
                    print(f"  ✓ {language}: {translated}")
            else:
                # If the column doesn't exist or is empty, keep it as is
                print(f"  Skipping empty column at index {col_idx}")
        
        # Write the row to the output file
        with open(output_file, 'a', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(new_row)
        
        # Save checkpoint after each row
        create_checkpoint(language, row_idx, current_delay)
        
        # Print progress
        print(f"Row {row_idx+1}/{total_rows} for {language} completed and saved")
        print(f"Current delay between requests: {current_delay}s")
        
        # Delay before next row with jitter
        if row_idx < total_rows - 1:  # Don't wait after the last row
            delay_with_jitter = current_delay + random.uniform(0, JITTER)
            print(f"Waiting {delay_with_jitter:.1f} second(s) before next row...")
            time.sleep(delay_with_jitter)
    
    print(f"Completed all translations for {language}!")
    return output_file

def main():
    print(f"Starting translation of {INPUT_FILE}")
    print(f"Translating columns {COLUMNS_TO_TRANSLATE} to: {', '.join(LANGUAGES)}")
    print(f"Each language will have its own output file")
    
    # Setup Gemini model
    model = setup_gemini()
    
    # Process the CSV for each language sequentially
    completed_files = []
    for language in LANGUAGES:
        print(f"\n{'='*50}")
        print(f"STARTING TRANSLATIONS FOR {language}")
        print(f"{'='*50}")
        
        output_file = process_csv_for_language(INPUT_FILE, language, model)
        completed_files.append((language, output_file))
    
    # Print summary
    print("\nTRANSLATION SUMMARY:")
    print("="*50)
    for language, file in completed_files:
        print(f"{language}: {file}")
    
    print("\nAll translations completed successfully!")

if __name__ == "__main__":
    main()