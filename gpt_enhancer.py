import csv
import os
from openai import OpenAI
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration variables
INPUT_FILE = 'test.csv'
OUTPUT_FILE = 'test_out.csv'  # Will be auto-generated based on INPUT_FILE and COLUMN_NAME if None
PROMPT = "You are a cultural expert. Given a simple factual question, generate a reasoning-based version that requires cultural commonsense to answer. Avoid directly naming the answer. Include contextual or narrative clues instead. It is necessary to add a reference to the image of the cultural artifact like 'as referenced in the image'. DO NOT include any prefixes or labels like 'Question:', 'Transformed question:', or similar text in your response. Return ONLY the rewritten question without any additional text."
COLUMN_NAME = "Analogy Question"
API_KEY = ""  # Your OpenAI API key, or leave empty to use environment variable
# Batch processing settings
BATCH_SIZE = 5  # Number of questions to process in a batch
BATCH_DELAY = 2  # Delay in seconds between batches
 
def save_progress(file_path, rows):
    """
    Save current progress to the output file
    """
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Progress saved to {file_path}")

def process_csv_with_gpt(input_file, output_file, prompt, column_name, api_key):
    """
    Process a CSV file, send each refactored question to GPT-4o with the given prompt,
    and add the response to a new column.
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Read input CSV
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Check if the column already exists, if not add it
        if column_name not in header:
            header.append(column_name)
        
        column_index = header.index(column_name)
        rows.append(header)
        
        # Read all rows from input file
        all_rows = list(reader)
        total_rows = len(all_rows)
        processed_count = 0
        
        print(f"Found {total_rows} rows to process")
        
        for row in all_rows:
            # Ensure the row has enough columns
            while len(row) < len(header):
                row.append("")
            
            # Get the refactored question and answer from the row
            refactored_question = row[3]  # Assuming "Refactored question" is at index 3
            answer = row[8]  # Assuming "Answer" is at index 8
            
            if refactored_question and answer:
                try:
                    # Send request to GPT-4o with both question and answer as context
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": f"Question: {refactored_question}\nThe answer is: {answer}"}
                        ]
                    )
                    
                    # Extract the generated response
                    response_text = completion.choices[0].message.content
                    
                    # Add the response to the column
                    row[column_index] = response_text
                    
                    # Print progress
                    processed_count += 1
                    print(f"[{processed_count}/{total_rows}] Processed: {refactored_question}")
                    print(f"Answer reference: {answer}")
                    print(f"Response: {response_text[:100]}...")  # Print first 100 chars
                    
                    # Batch processing logic - save after each batch
                    if processed_count % BATCH_SIZE == 0:
                        rows.append(row)
                        save_progress(output_file, rows)
                        print(f"Completed batch of {BATCH_SIZE}. Pausing for {BATCH_DELAY} seconds...")
                        time.sleep(BATCH_DELAY)
                    else:
                        rows.append(row)
                        # Short pause between individual requests
                        time.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing question '{refactored_question}': {str(e)}")
                    # Still add the row, but without a generated response
                    rows.append(row)
            else:
                # If no refactored question or answer, still include the row
                rows.append(row)
    
    # Save any remaining progress
    save_progress(output_file, rows)
    print(f"Processing complete. All {processed_count} questions processed.")

def main():
    global OUTPUT_FILE
    
    # If output file not specified, create one based on the column name
    if not OUTPUT_FILE:
        base_name = os.path.splitext(INPUT_FILE)[0]
        OUTPUT_FILE = f"{base_name}_with_{COLUMN_NAME}.csv"
        
    # Get API key from environment if not provided
    api_key = API_KEY or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key not provided. Please set the OPENAI_API_KEY environment variable or update the API_KEY variable in the script.")
        return
    
    print(f"Processing file: {INPUT_FILE}")
    print(f"Adding/updating column: {COLUMN_NAME}")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    print(f"Using prompt: {PROMPT}")
    print(f"Processing in batches of {BATCH_SIZE} with {BATCH_DELAY}s delay between batches")
    print("Starting processing...")
    
    process_csv_with_gpt(INPUT_FILE, OUTPUT_FILE, PROMPT, COLUMN_NAME, api_key)

if __name__ == "__main__":
    main()