# Create a GPT completion dataset using prompts from webtext and the GPT API.

import json
import os
import argparse
import threading
# from time import sleep
# import random

import openai

from tools import progress_bar

# Load API key
with open('openai_api_key.txt', 'r') as f:
    openai.api_key = f.read()

parser = argparse.ArgumentParser()
parser.add_argument('prompt_file', type=str, help='path to the prompt file')
parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo', help='GPT model to use')
parser.add_argument('--max_tokens', type=int, default=200, help='max tokens to generate')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--temperature', type=float, default=0.4, help='set temperature of GPT model')
parser.add_argument('--role_desc', type=str, default='You are a scientist. Complete the paper abstract.', help='role description')

args = parser.parse_args()
prompt_file = args.prompt_file
gpt_model = args.gpt_model
max_tokens = args.max_tokens
batch_size = args.batch_size
temperature = args.temperature
role_desc = args.role_desc

print('GPT model: {}'.format(gpt_model))
print('Max tokens: {}'.format(max_tokens))
print('Temperature: {}'.format(temperature))

data_dir = os.path.dirname(prompt_file)
prompt_file_name = os.path.basename(prompt_file)
out_file_name = gpt_model + '_completion' + prompt_file_name[prompt_file_name.find('.'):]          # + '_completion_t_' + str(temperature)
out_file = os.path.join(data_dir, out_file_name)
print('Writing to {}'.format(out_file))

# Create a thread for each prompt
def chat_completion_thread(prompt, completion_list, index):
    # print('Thread {} started'.format(index))
    request = [{"role": "system", "content": role_desc},
                                   {"role": "user", "content": prompt}]
    # request = [{"role": "system", "content": "You are a text completion model. Create as long of a text completion as you can."},
    #                                {"role": "user", "content": prompt}]
    
    retry_count = 0
    while retry_count < 10:
        try:
            response_text = openai.ChatCompletion.create(
                model=gpt_model,
                messages=request,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )['choices'][0]['message']['content']
        
            completion_list[index] = response_text
            # if retry_count > 0:
            #     print('Retry {} successful.'.format(retry_count))
            return
        
        except Exception as e:
            if retry_count > 0:
                print('\nRetry {} failed.'.format(retry_count))
            # print('\nAPI ERROR:')
            # print(e)
            # print('Error on prompt: {}'.format(prompt))
            # print('Retrying...')
            retry_count += 1

    completion_list[index] = -1

# Open output file in append mode and find number of lines
out_f = open(out_file, 'a')
num_lines = sum(1 for line in open(out_file))
print('Found {} lines in {}'.format(num_lines, out_file))

# Load prompts
prompts = []
with open(prompt_file, 'r') as f:
    for line in f:
        prompts.append(json.loads(line)['prompt'].strip())

print('Loaded {} prompts'.format(len(prompts)))
total_prompts = len(prompts)

# Remove prompts that have already been completed
prompts = prompts[num_lines:]

# Get completions
if gpt_model == 'gpt-3.5-turbo':
    print('Chat mode')
    print('Role description: {}'.format(role_desc))
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        response_batch = [None] * len(batch)

        threads = []
        for index, sample in enumerate(batch):
            # Randomize start time to avoid overloading API
            # sleep(random.random() * 0.5)
            x = threading.Thread(target=chat_completion_thread, args=(sample, response_batch, index))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()
            # print('Thread {} finished'.format(index))

        # Check for errors in batch
        if -1 in response_batch:
            print('Error in batch. Exiting...')
            exit()

        # Write to file
        for j, response in enumerate(response_batch):
            out_f.write(json.dumps({'prompt': batch[j].strip(), 'completion': response.strip()}) + '\n')
        
        out_f.flush()

        # Wait for user input
        # input('Press enter to continue...')

        # Print progress
        done = (num_lines + i + len(batch)) / total_prompts

        print('   Completed {} prompts. Progress {}'.format(num_lines + i + len(batch), progress_bar(done)), end='\r')

else:
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        response_texts = openai.Completion.create(
            model=gpt_model,
            prompt=batch,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )['choices']

        # Write to file
        for j in range(len(response_texts)):
            out_f.write(json.dumps({'prompt': batch[j].strip(), 'completion': response_texts[j]['text'].strip()}) + '\n')
        
        out_f.flush()

        # Print progress
        done = (num_lines + i + len(batch)) / total_prompts

        print('   Completed {} prompts. Progress {}'.format(num_lines + i + len(batch), progress_bar(done)), end='\r')

print('')
out_f.close()