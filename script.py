import csv
import collections

# Path to the input CSV file
input_file = 'Corrected_Questions-testing.csv'

# Number of rows to extract per state - change this value to get more or fewer rows
ROWS_PER_STATE = 1

# Create a dictionary to store rows by state
state_rows = collections.defaultdict(list)

# Read the CSV file
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # Store the header row
    
    # Group rows by state
    for row in reader:
        if row and row[0].strip() and not row[0].startswith('Personalities'):  # Skip empty or invalid state names
            state = row[0]  # First column is state
            state_rows[state].append(row)

# States with fewer than required rows (problematic states)
problematic_states = []

# Output filename based on the number of rows
output_file = f'first_{ROWS_PER_STATE}_rows_by_state.csv'

# Write output with specified number of rows for each state
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)  # Write the header row
    
    # For each state, write up to the specified number of rows (only if it has enough rows)
    for state, rows in state_rows.items():
        if len(rows) >= ROWS_PER_STATE:
            for row in rows[:ROWS_PER_STATE]:
                writer.writerow(row)
        else:
            problematic_states.append((state, len(rows)))

print(f'Processed {len(state_rows)} valid states.')
print(f'Wrote data for {len(state_rows) - len(problematic_states)} states with at least {ROWS_PER_STATE} rows.')
print(f'Output saved to {output_file}')

# Print information about states with at least the required number of rows
for state, rows in state_rows.items():
    if len(rows) >= ROWS_PER_STATE:
        print(f'{state}: {len(rows)} total rows, first {ROWS_PER_STATE} extracted')

# Print problematic states
if problematic_states:
    print(f"\nThe following states don't have enough rows (less than {ROWS_PER_STATE}):")
    for state, count in problematic_states:
        print(f'{state}: only {count} rows available')