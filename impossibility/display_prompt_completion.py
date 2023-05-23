# Display a prompt and its completion from a given line in the completion dataset.

import json
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('completion_file', type=str, help='path to the completion file')
parser.add_argument('line_num', type=int, help='line number to display')

args = parser.parse_args()
completion_file = args.completion_file
line_num = args.line_num

# Load prompt and completion from line
with open(completion_file, 'r') as f:
    for i, line in enumerate(f):
        if i == line_num:
            json_dict = json.loads(line)
            break

prompt = json_dict['prompt']
completion = json_dict['completion']

print('\nPROMPT:\n\n{}'.format(prompt))
print('\n' + '-' * 100 + '\n')
print('COMPLETION:\n\n{}'.format(completion))