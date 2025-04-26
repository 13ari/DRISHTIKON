import csv
import os
import pandas as pd
import google.generativeai as genai
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration variables - modify these as needed
INPUT_FILE = 'Generated_Questions.csv'                             # Input CSV file
OUTPUT_DIR = 'translated'                           # Directory to store translated files
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')   # API key from .env file
MODEL = 'gemini-2.5-flash-preview'                  # Gemini model to use
SAVE_AFTER_EACH_ROW = True                          # Whether to save after each row

# Columns to translate (by index) - adjust these for your CSV
COLUMNS_TO_TRANSLATE = [2, 3]                       # Indices of columns to translate (0-based index)

# Target languages for translation 
LANGUAGES = [
    'Hindi', 
    'Punjabi',
    'Urdu',
    'Telugu',
    'Marathi'
]

# Batch settings
DELAY_BETWEEN_ROWS = 1  # Seconds to wait between processing rows

def setup_gemini():
    """Configure the Gemini API."""
    if not GEMINI_API_KEY:
        raise ValueError("No Gemini API key found. Set GEMINI_API_KEY in your .env file")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL)
    return model

def translate_text(model, text, target_language):
    """Translate text to target language using Gemini."""
    if not text or not text.strip():
        return ""
    
    try:
        prompt = f"Translate the following text to {target_language}. Only return the translated text, no additional explanations or notes:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error translating to {target_language}: {str(e)}")
        return f"ERROR: {str(e)}"

def create_output_file(input_file, languages):
    """Create the output CSV file with additional columns for translations."""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Generate output filename
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(OUTPUT_DIR, f"{base_filename}_translated.csv")
    
    # Read input CSV to get column names
    df = pd.read_csv(input_file)
    header = list(df.columns)
    
    # Add new columns for each language and column to translate
    original_cols = [header[col_idx] for col_idx in COLUMNS_TO_TRANSLATE]
    new_headers = header.copy()
    
    for lang in languages:
        for col in original_cols:
            new_headers.append(f"{col}_{lang}")
    
    # Write header to output file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(new_headers)
    
    return output_file, header

def process_csv(input_file, output_file, header, model, languages):
    """Process the CSV and translate specified columns."""
    # Read input data
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        all_rows = list(reader)
    
    total_rows = len(all_rows)
    print(f"Found {total_rows} rows to process")
    
    # Process each row
    for row_idx, row in enumerate(all_rows):
        # Create a new row with original data plus translations
        new_row = row.copy()
        
        print(f"Processing row {row_idx+1}/{total_rows}")
        
        # Translate each required column to each target language
        for col_idx in COLUMNS_TO_TRANSLATE:
            if col_idx < len(row) and row[col_idx]:
                original_text = row[col_idx]
                col_name = header[col_idx] if col_idx < len(header) else f"Column_{col_idx}"
                
                for lang in languages:
                    print(f"  Translating {col_name} to {lang}...")
                    translated = translate_text(model, original_text, lang)
                    new_row.append(translated)
                    print(f"  ✓ {lang}: {translated[:40]}..." if len(translated) > 40 else f"  ✓ {lang}: {translated}")
            else:
                # If the column doesn't exist or is empty, add empty values
                for _ in languages:
                    new_row.append("")
        
        # Write the row to the output file
        with open(output_file, 'a', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(new_row)
        
        # Print progress
        print(f"Row {row_idx+1}/{total_rows} completed and saved")
        
        # Delay to avoid rate limits
        if row_idx < total_rows - 1:  # Don't wait after the last row
            time.sleep(DELAY_BETWEEN_ROWS)

def main():
    print(f"Starting translation of {INPUT_FILE}")
    print(f"Translating columns {COLUMNS_TO_TRANSLATE} to: {', '.join(LANGUAGES)}")
    
    # Setup Gemini model
    model = setup_gemini()
    
    # Create output file with headers
    output_file, header = create_output_file(INPUT_FILE, LANGUAGES)
    print(f"Output will be saved to {output_file}")
    
    # Process the CSV
    process_csv(INPUT_FILE, output_file, header, model, LANGUAGES)
    
    print("Translation completed successfully!")

if __name__ == "__main__":
    main()