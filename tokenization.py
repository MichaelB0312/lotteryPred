import csv
import torchtext

# Assuming you have a CSV file named 'data.csv' with columns 'num1', 'num2', 'num3', 'num4', 'num5', 'num6'
data = []  # To store the tokenized numbers

# Open the CSV file and read the numbers
with open('./data/weak_balls.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        # Tokenize each number individually and store them as separate tokens
        tokenized_numbers = [str(row['1']), str(row['2']), str(row['3']), str(row['4']), str(row['5']), str(row['6'])]
        data.extend(tokenized_numbers)

# Tokenizer (treat each number as a token)
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# Tokenize the numbers
tokenized_data = tokenizer(" ".join(data))  # Join the numbers with spaces and then tokenize

# Print tokenized numbers
print((tokenized_data))
