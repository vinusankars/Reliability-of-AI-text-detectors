# Remove leading and trailing spaces from completions

import os
import json
import argparse
import re

# text = 'hello\n\n\n\n\n\nworld'
# print(text)
# print(re.sub('\s*\n\s*', ' ', text))
# exit()

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='name of json file to remove spaces from')
parser.add_argument('--nl', action='store_true', help='remove newlines')

args = parser.parse_args()

filename = args.filename
nl = args.nl

# Load json file
json_list = []
with open(filename, 'r') as f:
    for line in f:
        json_obj = json.loads(line)

        prompt = json_obj['prompt']
        completion = json_obj['completion']

        # Remove leading and trailing spaces
        completion = completion.strip()

        # Remove newlines if specified
        if nl:
            completion = re.sub('\s*\n\s*', ' ', completion)

        # Add to json list
        json_list.append({'prompt': prompt, 'completion': completion})

# Write to file
with open(filename, 'w') as f:
    for json_obj in json_list:
        f.write(json.dumps(json_obj) + '\n')

print('Wrote %d samples to %s' % (len(json_list), filename))
