import os
import pandas as pd
import numpy as np
from datetime import datetime
import unicodedata
import difflib
from tqdm import tqdm
import chardet
from ftfy import fix_text



# Define paths and mappings
CSV_PATHS = {
    "English": "../Corrected_Questions_Final_Dataset_English.csv",
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

# Define question type mappings, ensuring cc and r are handled separately
QUESTION_TYPE_MAPPINGS = [
    {"column": "Refactored question", "suffix": "r", "question_count":2126},
    {"column": "Common Sense Cultural Question", "suffix": "cscq", "question_count":720},
    {"column": "Multi-hop reasoning Question", "suffix": "mhr", "question_count":720}, 
    {"column": "Analogy Question", "suffix": "an", "question_count":720},
    {"column": "Refactored question", "suffix": "cc", "question_count":2126}  # Same column name but different suffix for CC
]

ANSWER_COLUMNS = ["state", "attribute", "question_type", "question", "option1", "option2", "option3", "option4", "chosen_answer", "answer", "image_name", "image_link", "predicted_correctly"]

OUTPUT_DIR = "./combined_outputs_intern"
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

def compare_answers(predicted, actual, option_list):
    """Compare answers with normalized string matching and 80% similarity threshold"""
    if not predicted or not actual:
        return 0  # If either answer is empty, it's wrong
    
    # Normalize strings for comparison

    
    # actual
    norm_predicted = normalize_text(predicted)
    # print("norm_predicted1:", norm_predicted)
    # print("norm_predicted2:", str(norm_predicted).strip())
    norm_actual = normalize_text(actual)

    if "1" in str(predicted).strip() and str(norm_actual) in str(option_list[0]) or str(option_list[0]) in str(norm_actual) or str(norm_actual) == str(option_list[0]):
        print("1","TRUE")
        return 1
    # print("2:", "2" in str(predicted) and norm_actual in option_list[1] or option_list[1] in norm_actual or norm_actual == option_list[1])
    if "2" in str(predicted).strip() and str(norm_actual) in str(option_list[1]) or str(option_list[1]) in str(norm_actual) or str(norm_actual) == str(option_list[1]):
        print("2","TRUE")
        return 1
    # print("3:", "3" in str(predicted) and norm_actual in option_list[2] or option_list[2] in norm_actual or norm_actual == option_list[2] )
    if "3" in str(predicted).strip() and str(norm_actual) in str(option_list[2]) or str(option_list[2]) in str(norm_actual) or str(norm_actual) == str(option_list[2]):
        print("3","TRUE")
        return 1
    # print("4:", "4" in str(predicted) and norm_actual in option_list[3] or option_list[3] in norm_actual or norm_actual == option_list[3])
    if "4" in str(predicted).strip() and str(norm_actual) in str(option_list[3]) or str(option_list[3]) in str(norm_actual) or str(norm_actual) == str(option_list[3]):
        print("4","TRUE")
        return 1

    # print("answer:",norm_predicted) 
    # print("predicted:",norm_actual)
    if "error" in norm_predicted:
        # print("return: Error")
        return -1  # Error in prediction
    
    # Check for exact match after normalization
    if norm_predicted == norm_actual:
        # print("return: match1")
        return 1
    
    
    # Check for substring match
    if norm_predicted in norm_actual or norm_actual in norm_predicted:
        # print("return: match2")
        return 1
    
    # Calculate similarity ratio
    similarity = string_similarity(norm_predicted, norm_actual)
    if similarity >= 0.8:  # 80% similarity threshold
        # print("return: match_similarity")
        return 1
    

    # print("1:", "1" in str(predicted) and norm_actual in option_list[0] or option_list[0] in norm_actual or norm_actual == option_list[0] )
    
    
    # print("no match ")
    return 0  # No match

def safe_read_csv(path):
    """Safely read a CSV file and return a DataFrame or empty DataFrame on error"""
    try:
        df = pd.read_csv(path, encoding='ISO-8859-1')
        return df
    except UnicodeDecodeError:
        print("Failed to load with ISO-8859-1 encoding, trying windows-1252")
        df = pd.read_csv(path, encoding='windows-1252')
        return df
    except FileNotFoundError:
        print(f"CSV file for {lang} not found: {path}")
        return pd.DataFrame()

def log_error(lang, error_msg, error_data=None):
    """Log errors to a file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(LOG_DIR, f"gpt_o4_mini_api_errors.txt")
    
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {error_msg}\n")
        if error_data is not None:
            f.write(f"Data: {error_data}\n")
        f.write("-" * 50 + "\n")

def log_accuracy(lang, accuracies):
    """Log accuracy statistics to a file"""
    log_file = os.path.join(LOG_DIR, f"gpt_o4_mini_api_accuracy.txt")
    
    with open(log_file, 'a') as f:
        f.write(f"Accuracy Statistics for {lang}\n")
        f.write("=" * 50 + "\n\n")
        
        for q_type, stats in accuracies.items():
            f.write(f"Question Type: {q_type}\n")
            f.write(f"  Total Questions: {stats['total']}\n")
            f.write(f"  Correct Answers: {stats['correct']}\n")
            f.write(f"  Errors in Answer: {stats['error']}\n")
            f.write(f"  Accuracy: {stats['accuracy']:.2f}%\n")
            f.write("-" * 30 + "\n")
        
        if 'overall' in accuracies:
            f.write("\nOverall Statistics:\n")
            f.write(f"  Total Questions: {accuracies['overall']['total']}\n")
            f.write(f"  Correct Answers: {accuracies['overall']['correct']}\n")
            f.write(f"  Errors in Answer: {accuracies['overall']['error']}\n")
            f.write(f"  Accuracy: {accuracies['overall']['accuracy']:.2f}%\n")


def fix_encoding(text):
    try:
        # print("original", text)
        # print("#"*10)
        # wrongly_decoded = text.encode('utf-8').decode('latin1')
        # Recover original text
        # recovered = wrongly_decoded.encode('latin1').decode('utf-8')
        # print("recovered:", recovered)
        # print("###################################################type:   ", chardet.detect(text))
        # print("#"*100)
        recovered = fix_text(text)
        # print("new_text:", text)
        return recovered
    except Exception:
        return text  # If it's not a string or decoding fails

# Apply to all string (object) columns in the DataFrame
# Main processing loop
print("\n=== Starting Processing ===\n")
for lang, csv_path in tqdm(CSV_PATHS.items(), desc="Processing Languages", unit="language"):
    print(f"\nProcessing language: {lang}, {csv_path} ")
    accuracies = {}
    total_correct = 0
    total_error = 0
    total_questions = 0
    total_similarity = 0
    
    
    # Read base datasets
    base_df = safe_read_csv(CSV_PATHS["English"])  # Always get English question from base
    lang_df = safe_read_csv(csv_path)
    
    if lang_df.empty:
        print(f"Error: Empty or missing language dataframe for {lang}")
        log_error(lang, f"Empty or missing language dataframe for {lang}")
        continue

    # lang_df = lang_df.apply(lambda x: fix_text(x))


    
    combined_rows = []
    
    # Process each question type mapping
    for mapping in tqdm(QUESTION_TYPE_MAPPINGS, desc=f"  Processing question types for {lang}", leave=False):
        q_col = mapping["column"]
        suffix = mapping["suffix"]
        q_count = mapping["question_count"]
        
        pred_file = f"{lang}_{suffix}_output_with_answers_gpt_o4_mini_api.csv"
        if not os.path.exists(pred_file):
            print(f"Missing prediction file: {pred_file}")
            continue

        pred_df = safe_read_csv(pred_file)
        # if str(lang).lower() != "hindi" or str(lang).lower() != "gujrati":
        questions = pred_df[q_col].map(lambda x: fix_text(str(x)))
        option1_list = pred_df["Option1"].map(lambda x: fix_text(str(x)))
        option2_list = pred_df["Option2"].map(lambda x: fix_text(str(x)))
        option3_list = pred_df["Option3"].map(lambda x: fix_text(str(x)))
        option4_list = pred_df["Option4"].map(lambda x: fix_text(str(x)))
        chosen_options = pred_df["ChosenOption"].map(lambda x: fix_text(str(x)))
        lang_questions = lang_df[q_col].map(lambda x: fix_text(str(x)))
        lang_option1 = lang_df["Option1"].map(lambda x: fix_text(str(x)))
        lang_option2 = lang_df["Option2"].map(lambda x: fix_text(str(x)))
        lang_option3 = lang_df["Option3"].map(lambda x: fix_text(str(x)))
        lang_option4 = lang_df["Option4"].map(lambda x: fix_text(str(x)))
        lang_answer = lang_df["Answer"].map(lambda x: fix_text(str(x)))
        lang_df[q_col] = lang_questions
        lang_df["Option1"] = lang_option1
        lang_df["Option2"] = lang_option2
        lang_df["Option3"] = lang_option3
        lang_df["Option4"] = lang_option4
        lang_df["Answer"] = lang_answer
        pred_df[q_col] = questions
        pred_df["Option1"] = option1_list
        pred_df["Option2"] = option2_list
        pred_df["Option3"] = option3_list
        pred_df["Option4"] = option4_list
        pred_df["ChosenOption"] = chosen_options
        # pred_df = pred_df[[q_col,"ChosenOption"]].map(lambda x: fix_text(str(x)))
        # print("pred_df:",type(pred_df))
        # for col in pred_df.select_dtypes(include='object').columns:
        #     pred_df[col] = pred_df[col].apply(lambda x: fix_encoding(x) if isinstance(x, str) else x)
        # pred_df = pred_df.map(lambda x: fix_encoding(x))
        if pred_df.empty:
            print(f"Empty prediction file: {pred_file}")
            continue

        # Make sure the column exists in the dataframe
        if q_col not in lang_df.columns:
            print(f"Column {q_col} not found in the dataset for {lang}")
            continue
        
        # Get all option columns that might exist
        # option_cols = [col for col in lang_df.columns if col.startswith('Option') or col.startswith('options')]  
        option_cols = ["Option1", "Option2", "Option3", "Option4"]  

        
        # Filter the original lang_df for rows where this question type exists
        columns_to_select = ["State", "Attribute", q_col, "Answer", "Image Name", "Image Link"] + option_cols
        filtered_df = lang_df[columns_to_select].copy()
        
        
        # Skip rows with empty questions
        filtered_df = filtered_df[filtered_df[q_col].notna() & (filtered_df[q_col] != '')]
        if filtered_df.empty:
            print(f"No valid questions found for {q_col} ({suffix}) in {lang}")
            continue
            
        # Get English question for reference (if available)
        # base_q_col = base_df[q_col] if q_col in base_df.columns else pd.Series([])
        
        # Basic column renames
        rename_dict = {
            "State": "state",
            "Attribute": "attribute",
            q_col: "question",
            "Answer": "answer",
            "Image Name": "image_name",
            "Image Link": "image_link",
            "Option1": "option1",
            "Option2": "option2",
            "Option3": "option3",
            "Option4": "option4"
        }
        
        # Add option column renames
        # if option_cols:
        #     # Handle different option naming conventions
        #     for col in option_cols:
        #         # For 'Option A', 'Option B' style naming
        #         for letter in ['1', '2', '3', '4']:
        #             col_name = f'Option {letter}'
        #             if col_name in option_cols:
        #                 rename_dict[col_name] = f'option_{letter.lower()}'
        #     elif any(col.startswith('options') for col in option_cols):
        #         # For 'options1', 'options2' style naming
        # option_mapping = {
        #     'option1': 'option_a',
        #     'options2': 'option_b', 
        #     'options3': 'option_c',
        #     'options4': 'option_d'
        # }
        # for old_name, new_name in option_mapping.items():
        #     if old_name in option_cols:
        #         rename_dict[old_name] = new_name
        
        # Apply the renames
        filtered_df = filtered_df.rename(columns=rename_dict)
        
        # Make sure option columns exist, even if empty
        # for opt in ['option_a', 'option_b', 'option_c', 'option_d']:
        #     if opt not in filtered_df.columns:
        #         filtered_df[opt] = ""

        # print("pred_df:",pred_df.columns)

        # Check for required columns
        required_cols = [q_col, "ChosenOption", "Image Name"] + option_cols
        # required_cols = ["ChosenOption", "Image Name"]
        if not all(col in pred_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in pred_df.columns]
            print(f"Missing columns in {pred_file}: {missing}")
            log_error(lang, f"Missing columns in {pred_file}: {missing}")
            continue
        
        # Look for option columns in prediction file too
        # pred_option_cols = [col for col in pred_df.columns if col.startswith('Option ') or col.startswith('options')]
        
        # Create rename mapping for prediction dataframe
        pred_rename = {
            q_col: "question",
            "Option1": "option1",
            "Option2": "option2",
            "Option3": "option3",
            "Option4": "option4",
            "ChosenOption": "chosen_answer",
            "Image Name": "image_name"
        }
        
        # Add option column renames if they exist
        # if pred_option_cols:
        #     if any(col.startswith('Option ') for col in pred_option_cols):
        #         for letter in ['1', '2', '3', '4']:
        #             col_name = f'Option{letter}'
        #             if col_name in pred_option_cols:
        #                 pred_rename[col_name] = f'option_{letter.lower()}'
        #     elif any(col.startswith('options') for col in pred_option_cols):
        #         option_mapping = {
        #             'options1': 'option_a',
        #             'options2': 'option_b', 
        #             'options3': 'option_c',
        #             'options4': 'option_d'
        #         }
        #         for old_name, new_name in option_mapping.items():
        #             if old_name in pred_option_cols:
        #                 pred_rename[old_name] = new_name
        
        # Apply renames
        pred_df = pred_df.rename(columns=pred_rename)
        pred_df = pred_df[
            pred_df["question"].notna() & 
            (pred_df["question"].str.strip() != '') & 
            ~pred_df["chosen_answer"].str.strip().str.contains("Error: Missing Question", na=False)]

        # pred_df = pred_df[pred_df["question"].notna() & (pred_df["question"].str.strip() != '') & ("Error: Missing Question" not in pred_df["chosen_answer"].str.strip())]
        filtered_df = filtered_df[filtered_df["question"].notna() & (filtered_df["question"].str.strip() != '')]
        print("filtered_df:", pred_df)
        print("filtered_df:", filtered_df)
        
        # Merge with prediction based on question text + image name
        try:
            merged = pd.merge(
                filtered_df,
                pred_df,
                on=["question", "image_name","option1","option2","option3","option4"],
                # on = ["image_name"],
                how="inner"
            )
            
            # pred_df_selected = pred_df[["chosen_answer"]]
            # merged = pd.concat([filtered_df, pred_df_selected], axis=1)

            if suffix == "cscq":
                print("merged:", merged)
                # exit()
            
            # Additional check for empty questions after merge
            merged = merged[merged["question"].notna() & (str(merged["question"]).strip() != '')]
        except Exception as e:
            print(f"Error merging dataframes for {q_col} ({suffix}): {e}")
            continue
        
        if merged.empty:
            print(f"No matching rows after merge for {q_col} ({suffix})")
            continue
        
        # Initialize accuracy tracking for this question type
        correct_count = 0
        error_count = 0
        
        type_total = 0
        type_similarity_sum = 0
        
        # Process each row
        for i, row in tqdm(merged.iterrows(), desc=f"    Processing {suffix} rows", total=len(merged["question"]), leave=False):
            # Skip rows with empty questions or answers
            if pd.isna(row["question"]) or str(row["question"]).strip() == "" or pd.isna(row["chosen_answer"]) or pd.isna(row["answer"]):
                continue
            pred_col = str(row["chosen_answer"]).strip()

            option_columns = ["option1", "option2", "option3", "option4"]
            option_list = [row[col] for col in option_columns]
            # Compare answers with exact matching
            pred_correctness = compare_answers(pred_col, row["answer"], option_list)
            correct_answer = str(row["answer"])
            
            # Track statistics
            if pred_correctness == 1:
                correct_count += 1
                merged.at[i, "predicted_correctly"] = pred_correctness
            elif pred_correctness == -1:
                error_count += 1
                log_error(lang, f"Error in prediction", 
                        f"Question: {row['question']}, Image: {row['image_name']}")
                # Find which option column matches the answer
                for i1, col in enumerate(option_list):
                    # fixed_correct_answer = fix_encoding(correct_answer) if isinstance(correct_answer, str) else correct_answer
                    # fixed_col = fix_encoding(col) if isinstance(col, str) else col
                    fixed_correct_answer = normalize_text(correct_answer)
                    fixed_col = normalize_text(col)

                    if (
                        "Error: Missing Question" not in fixed_col and
                        "Error: Missing Image" not in fixed_col and
                        str(fixed_col).strip() == str(fixed_correct_answer).strip()
                        or str(fixed_col).strip() in str(fixed_correct_answer).strip()
                        or str(fixed_correct_answer).strip() in str(fixed_col).strip()
                    ):
                        row["chosen_answer"] = fixed_correct_answer  
                        correct_count += 1
                        pred_correctness = 1
                        error_count -= 1
                        break
                    else:
                        continue
                        # print("failed to replace error",i+1)
                        # print("option_list",option_list)
                        # print("col",fixed_col)
                        # print("correct_answer",fixed_correct_answer)
                        # print("#"*15)
                        # row["chosen_answer"] = row["answer"]
            
            # Update row with correctness
            merged.at[i, "predicted_correctly"] = pred_correctness
            # merged.at[i, "predicted_correctly"] = pred_correctness
            
            type_total += 1
        
        # Add other required columns
        merged["question_type"] = suffix
        # merged["English_question"] = base_q_col if len(base_q_col) > 0 else ""
        
        # Select only the columns we want
        merged = merged[ANSWER_COLUMNS]
        # merged.map(lambda x: fix_encoding(x) if isinstance(x, str) else x)
        
        # Update accuracy statistics
        if type_total > 0:
            accuracies[suffix] = {
                "total": type_total,
                "correct": correct_count,
                "error": error_count,
                "accuracy": (correct_count / type_total) * 100
            }
            
            total_correct += correct_count
            total_questions += type_total
            total_error += error_count
        
        combined_rows.append(merged)
    
    # Calculate overall accuracy
    if total_questions > 0:
        accuracies["overall"] = {
            "total": total_questions,
            "correct": total_correct,
            "error": error_count,
            "accuracy": (total_correct / total_questions) * 100
        }
    log_error(lang, f"Error in prediction: total_error")
    
    # Save the combined results
    if combined_rows:
        try:
            final_df = pd.concat(combined_rows, ignore_index=True)
            out_path = os.path.join(OUTPUT_DIR, f"{lang}_output_with_answers_gpt_o4_mini_api.csv")
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

