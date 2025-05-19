import os
import pandas as pd
import numpy as np
from datetime import datetime
import unicodedata
import difflib
from tqdm import tqdm

# Define paths and mappings
CSV_PATHS = {
    "English": "../Corrected_Questions_Final_Dataset_English.csv",
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

# Define question type mappings, ensuring cc and r are handled separately
QUESTION_TYPE_MAPPINGS = [
    {"column": "Refactored question", "suffix": "r"},
    {"column": "Common Sense Cultural Question", "suffix": "cscq"},
    {"column": "Multi-hop reasoning Question", "suffix": "mhr"}, 
    {"column": "Analogy Question", "suffix": "an"},
    {"column": "Refactored question", "suffix": "cc"}  # Same column name but different suffix for CC
]

ANSWER_COLUMNS = ["state", "attribute", "question_type", "question", "option_a", "option_b", "option_c", "option_d", "chosen_answer", "answer", "image_name", "image_link", "predicted_correctly"]

OUTPUT_DIR = "./combined_outputs"
LOG_DIR = "./error_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def normalize_text(text):
    """Normalize text for consistent comparison"""
    if not text:
        return ""
    
    # Convert to string and normalize unicode
    text = str(text).strip().lower()
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def string_similarity(s1, s2):
    """Calculate the similarity ratio between two strings"""
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def compare_answers(predicted, actual):
    """Compare answers with normalized string matching and 80% similarity threshold"""
    if not predicted or not actual:
        return 0  # If either answer is empty, it's wrong
    
    # Normalize strings for comparison
    norm_predicted = normalize_text(predicted)
    norm_actual = normalize_text(actual)
    
    if "error" in norm_predicted:
        return -1  # Error in prediction
    
    # Check for exact match after normalization
    if norm_predicted == norm_actual:
        return 1
    
    # Check for substring match
    if norm_predicted in norm_actual or norm_actual in norm_predicted:
        return 1
    
    # Calculate similarity ratio
    similarity = string_similarity(norm_predicted, norm_actual)
    if similarity >= 0.8:  # 80% similarity threshold
        return 1
    
    return 0  # No match

def safe_read_csv(path):
    """Safely read a CSV file and return a DataFrame or empty DataFrame on error"""
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame()

def log_error(lang, error_msg, error_data=None):
    """Log errors to a file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(LOG_DIR, f"{lang}_errors.txt")
    
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {error_msg}\n")
        if error_data is not None:
            f.write(f"Data: {error_data}\n")
        f.write("-" * 50 + "\n")

def log_accuracy(lang, accuracies):
    """Log accuracy statistics to a file"""
    log_file = os.path.join(LOG_DIR, f"{lang}_accuracy.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"Accuracy Statistics for {lang}\n")
        f.write("=" * 50 + "\n\n")
        
        for q_type, stats in accuracies.items():
            f.write(f"Question Type: {q_type}\n")
            f.write(f"  Total Questions: {stats['total']}\n")
            f.write(f"  Correct Answers: {stats['correct']}\n")
            f.write(f"  Accuracy: {stats['accuracy']:.2f}%\n")
            f.write("-" * 30 + "\n")
        
        if 'overall' in accuracies:
            f.write("\nOverall Statistics:\n")
            f.write(f"  Total Questions: {accuracies['overall']['total']}\n")
            f.write(f"  Correct Answers: {accuracies['overall']['correct']}\n")
            f.write(f"  Accuracy: {accuracies['overall']['accuracy']:.2f}%\n")

# Main processing loop
print("\n=== Starting Processing ===\n")
for lang, csv_path in tqdm(CSV_PATHS.items(), desc="Processing Languages", unit="language"):
    print(f"\nProcessing language: {lang}")
    accuracies = {}
    total_correct = 0
    total_questions = 0
    total_similarity = 0
    
    # Read base datasets
    base_df = safe_read_csv(CSV_PATHS["English"])  # Always get English question from base
    lang_df = safe_read_csv(csv_path)
    
    if lang_df.empty:
        print(f"Error: Empty or missing language dataframe for {lang}")
        log_error(lang, f"Empty or missing language dataframe for {lang}")
        continue
    
    combined_rows = []
    
    # Process each question type mapping
    for mapping in tqdm(QUESTION_TYPE_MAPPINGS, desc=f"  Processing question types for {lang}", leave=False):
        q_col = mapping["column"]
        suffix = mapping["suffix"]
        
        pred_file = f"{lang}_{suffix}_output_with_answers_qwenvl.csv"
        if not os.path.exists(pred_file):
            print(f"Missing prediction file: {pred_file}")
            continue

        pred_df = safe_read_csv(pred_file)
        if pred_df.empty:
            print(f"Empty prediction file: {pred_file}")
            continue

        # Make sure the column exists in the dataframe
        if q_col not in lang_df.columns:
            print(f"Column {q_col} not found in the dataset for {lang}")
            continue
        
        # Get all option columns that might exist
        option_cols = [col for col in lang_df.columns if col.startswith('Option ') or col.startswith('options')]  
        
        # Filter the original lang_df for rows where this question type exists
        columns_to_select = ["State", "Attribute", q_col, "Answer", "Image Name", "Image Link"] + option_cols
        filtered_df = lang_df[columns_to_select].copy()
        
        # Skip rows with empty questions
        filtered_df = filtered_df[filtered_df[q_col].notna() & (filtered_df[q_col] != '')]
        if filtered_df.empty:
            print(f"No valid questions found for {q_col} ({suffix}) in {lang}")
            continue
            
        # Get English question for reference (if available)
        base_q_col = base_df[q_col] if q_col in base_df.columns else pd.Series([])
        
        # Basic column renames
        rename_dict = {
            "State": "state",
            "Attribute": "attribute",
            q_col: "question",
            "Answer": "answer",
            "Image Name": "image_name",
            "Image Link": "image_link"
        }
        
        # Add option column renames
        if option_cols:
            # Handle different option naming conventions
            if any(col.startswith('Option ') for col in option_cols):
                # For 'Option A', 'Option B' style naming
                for letter in ['A', 'B', 'C', 'D']:
                    col_name = f'Option {letter}'
                    if col_name in option_cols:
                        rename_dict[col_name] = f'option_{letter.lower()}'
            elif any(col.startswith('options') for col in option_cols):
                # For 'options1', 'options2' style naming
                option_mapping = {
                    'options1': 'option_a',
                    'options2': 'option_b', 
                    'options3': 'option_c',
                    'options4': 'option_d'
                }
                for old_name, new_name in option_mapping.items():
                    if old_name in option_cols:
                        rename_dict[old_name] = new_name
        
        # Apply the renames
        filtered_df = filtered_df.rename(columns=rename_dict)
        
        # Make sure option columns exist, even if empty
        for opt in ['option_a', 'option_b', 'option_c', 'option_d']:
            if opt not in filtered_df.columns:
                filtered_df[opt] = ""
        
        # Check for required columns
        required_cols = [q_col, "ChosenOption", "Image Name"]
        if not all(col in pred_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in pred_df.columns]
            print(f"Missing columns in {pred_file}: {missing}")
            log_error(lang, f"Missing columns in {pred_file}: {missing}")
            continue
        
        # Look for option columns in prediction file too
        pred_option_cols = [col for col in pred_df.columns if col.startswith('Option ') or col.startswith('options')]
        
        # Create rename mapping for prediction dataframe
        pred_rename = {
            q_col: "question",
            "ChosenOption": "chosen_answer",
            "Image Name": "image_name"
        }
        
        # Add option column renames if they exist
        if pred_option_cols:
            if any(col.startswith('Option ') for col in pred_option_cols):
                for letter in ['A', 'B', 'C', 'D']:
                    col_name = f'Option {letter}'
                    if col_name in pred_option_cols:
                        pred_rename[col_name] = f'option_{letter.lower()}'
            elif any(col.startswith('options') for col in pred_option_cols):
                option_mapping = {
                    'options1': 'option_a',
                    'options2': 'option_b', 
                    'options3': 'option_c',
                    'options4': 'option_d'
                }
                for old_name, new_name in option_mapping.items():
                    if old_name in pred_option_cols:
                        pred_rename[old_name] = new_name
        
        # Apply renames
        pred_df = pred_df.rename(columns=pred_rename)
        
        # Merge with prediction based on question text + image name
        try:
            merged = pd.merge(
                filtered_df,
                pred_df,
                on=["question", "image_name"],
                how="inner"
            )
            
            # Additional check for empty questions after merge
            merged = merged[merged["question"].notna() & (merged["question"] != '')]
        except Exception as e:
            print(f"Error merging dataframes for {q_col} ({suffix}): {e}")
            continue
        
        if merged.empty:
            print(f"No matching rows after merge for {q_col} ({suffix})")
            continue
        
        # Initialize accuracy tracking for this question type
        correct_count = 0
        type_total = 0
        type_similarity_sum = 0
        
        # Process each row
        for i, row in tqdm(merged.iterrows(), desc=f"    Processing {suffix} rows", total=len(merged), leave=False):
            # Skip rows with empty questions or answers
            if pd.isna(row["question"]) or str(row["question"]).strip() == "" or pd.isna(row["chosen_answer"]) or pd.isna(row["answer"]):
                continue
                
            # Compare answers with exact matching
            pred_correctness = compare_answers(row["chosen_answer"], row["answer"])
            
            # Track statistics
            if pred_correctness == 1:
                correct_count += 1
            elif pred_correctness == -1:
                log_error(lang, f"Error in prediction", 
                          f"Question: {row['question']}, Image: {row['image_name']}")
            
            # Update row with correctness
            merged.at[i, "predicted_correctly"] = pred_correctness
            
            type_total += 1
        
        # Add other required columns
        merged["question_type"] = suffix
        merged["English_question"] = base_q_col if len(base_q_col) > 0 else ""
        
        # Select only the columns we want
        merged = merged[ANSWER_COLUMNS]
        
        # Update accuracy statistics
        if type_total > 0:
            accuracies[suffix] = {
                "total": type_total,
                "correct": correct_count,
                "accuracy": (correct_count / type_total) * 100
            }
            
            total_correct += correct_count
            total_questions += type_total
        
        combined_rows.append(merged)
    
    # Calculate overall accuracy
    if total_questions > 0:
        accuracies["overall"] = {
            "total": total_questions,
            "correct": total_correct,
            "accuracy": (total_correct / total_questions) * 100
        }
    
    # Save the combined results
    if combined_rows:
        try:
            final_df = pd.concat(combined_rows, ignore_index=True)
            out_path = os.path.join(OUTPUT_DIR, f"{lang}_output_with_answers_qwenvl.csv")
            final_df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")
            
            # Log accuracy statistics
            log_accuracy(lang, accuracies)
            
        except Exception as e:
            print(f"Error saving final dataframe for {lang}: {e}")
            log_error(lang, f"Error saving final dataframe: {e}")
    else:
        print(f"No valid data to combine for language: {lang}")

print("\nProcessing complete. Check the combined_outputs directory for results and error_logs for any issues.")