import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
# from transformers import pipeline, AutoTokenizer
import torch
import time
import gc
import re
import os

try:
    data = pd.read_csv(r"Missing_Dataset_Questions.csv", encoding='ISO-8859-1')
except UnicodeDecodeError:
    print("Failed to load with ISO-8859-1 encoding, trying windows-1252")
    data = pd.read_csv(r"Missing_Dataset_Questions.csv", encoding='windows-1252')

gc.collect()  # Run garbage collection to free CPU RAM
torch.cuda.empty_cache()  # Clear GPU memory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")



start_time = time.time()

# Configure BitsAndBytesConfig for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)


model_name = "unsloth/Qwen2.5-72B-bnb-4bit"
huggingface_token = os.environ.get('HF_API_TOKEN')

tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)

print("Loading the model and pipeline across GPUs...")

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config,
    token=huggingface_token).to(device)

print("Model and tokenizer loaded successfully")

# Step 4: Process the dataset
progress = 0
correct_answers = 0
total_questions = len(data)
model_answers = []
predicted_correctly = []

for index, row in data.iterrows():
    # Define the question prompt with options
    question_with_options = (
        f"Answer the following question by selecting the correct option from the options listed below. "
        f"Respond with the full answer choice (e.g., 'C) Indian') and nothing else. "
        f"Do not include additional questions, explanations, or text."
        f"\nQuestion: {row['Question']} "
        f"Options: "
        f"A) {row['Option1']} "
        f"B) {row['Option2']} "
        f"C) {row['Option3']} "
        f"D) {row['Option4']}"
    )


    # Tokenize the input
    inputs = tokenizer.encode(question_with_options, return_tensors="pt").to(device)

    # Generate the answer from the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=200)

    # Decode the output tokens to get the answer text
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process the model's answer to match the format of the correct answer
    model_answer = model_answer.replace(question_with_options, "").strip()
   
    # Post-process the model's answer to validate correctness
    if row['Answer'].strip().lower() in model_answer.strip().lower():
        prediction_correctness = 'True'
        correct_answers += 1
    else:
        prediction_correctness = 'False'

    progress += 1
    print(f"Progress: {progress}/{total_questions}")
    print(f"Model Answer: {model_answer}")
    print(f"Correct Answer: {row['Answer']}")

    # Append results for analysis
    model_answers.append(model_answer)
    predicted_correctly.append(prediction_correctness)
    print("Correct answers so far:", correct_answers)

# Calculate the accuracy
accuracy = correct_answers / total_questions
print(f"Final Accuracy: {accuracy * 100:.2f}%")
print(f"Total Correct Answers: {correct_answers}")

# Step 5: Save results to a CSV file
data['Model_Answer'] = model_answers
data['Predicted Correctly'] = predicted_correctly
data.to_csv('Missing_Dataset_Questions_qwen2_5_72b.csv', index=False)

# Step 6: End the timer
end_time = time.time()
time_taken = end_time - start_time

print("Time taken:", time_taken, "seconds")
print("Khatam tata goodbye")

