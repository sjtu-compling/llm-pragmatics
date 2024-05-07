from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import sys
import argparse
import pandas as pd
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_id", type=str, default="gpt2")
parser.add_argument("--expl_dir", type=str, default="expl")
parser.add_argument("--style", choices=['verbose', 'concise'], default="concise", help="choose verbose to have responses kept in txt files.")
parser.add_argument("--legacy", action="store_true", default=False, help="Whether to use legacy tokenizer.(reported when using Chinese alpaca model)")
parser.add_argument("--index", action="store_true", default=False, help="Whether to name the response files with indices of questions in the original form.")
parser.add_argument("--max_new_tokens", type=int, default=50, help="The maximum number of tokens to generate.")
args = parser.parse_args()

if not os.path.exists(args.expl_dir):
    os.mkdir(args.expl_dir)

expl_dir = args.expl_dir
model_id = args.model_id
model_name = model_id.split("/")[-1]
if not os.path.exists(f"{expl_dir}/{model_name}"):
    os.makedirs(f"{expl_dir}/{model_name}")


model = AutoModelForCausalLM.from_pretrained(model_id).cpu()
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=args.legacy)

df = pd.read_csv(sys.stdin)
inputs = df['Question']
csv_file = f"{model_name}_response.csv"
if csv_file not in os.listdir(f"{expl_dir}/{model_name}"):
    with open(f"{expl_dir}/{model_name}/{csv_file}", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Response', 'maxim'])
for i, input in enumerate(inputs):
    # Embedding
    prompt = "你现在是一个中文母语者。" + input
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.cpu()
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_new_tokens=args.max_new_tokens, # 512 for text generate, 50 for choice
        temperature=0.9,
        top_k=3,  # 0 for text generate, 3 for choice
        top_p=0.9, # 0.9 for text generate, 0.1 for choice 
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1, 
    )
    response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    # save the response to a file
    if args.index:
        idx = df['Index'][i]
    else:
        idx = i + 1
    if args.style == 'verbose':
        with open(f"{expl_dir}/{model_name}/{model_name}_response_{idx}.txt", "w") as f:
            f.write(f"Qustion{idx}: {input}")
            f.write(f"Response{idx}: {response}")
        print(f"Response {idx} saved to response_{idx}.txt and {model_name}_response.csv")
    else:
        print(f"Adding Response {idx} to {model_name}_response.csv")
    with open(f"{expl_dir}/{model_name}/{model_name}_response.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([idx, response, df['maxim'][i]])