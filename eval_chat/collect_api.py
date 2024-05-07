import openai
import json
import pandas as pd
import argparse
import os
from requests.exceptions import Timeout
import google.generativeai as genai
import sys
import csv

def get_collection_responses(prompt_, model, temp):
    for _ in range(MAX_RETIRES):
        try:
            response = openai.Completion.create(
                model=model,
                prompt=prompt_,
                temperature=temp,
            )
            break
        except Timeout:
            print("Request timed out. Retrying...")
        except Exception as e:
            print(f"An error occurred: {e}")
            break  # Handle other exceptions as needed
    return response['choices'][0]['text']

def get_gemini(message, model, temp):
    for _ in range(MAX_RETIRES):
        try:
            response = model.generate_content(
                message
            )
            break
        except Timeout:
            print("Request timed out. Retrying...")
        except Exception as e:
            print(f"An error occurred: {e}")
            break  # Handle other exceptions as needed
    return response.parts

def get_Chat_responses(message, model, temp):
    for _ in range(MAX_RETIRES):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=message,
                temperature=temp,
            )
            break
        except Timeout:
            print("Request timed out. Retrying...")
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return response['choices'][0]['message']['content']


if __name__ == "__main__":
    legacy_models = ["text-davinci-003", "babbage-002", "davinci-002", "text-davinci-002", "davinci", "curie", "ada", "babbage"]
    newer_models = ["gpt-4", "gpt-3.5-turbo", 'default-model']

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--style', choices=['verbose', 'concise'], default="concise", help="choose verbose to have responses kept in txt files.")
    argparser.add_argument('-m', '--model', type=str, default="gpt-4")
    argparser.add_argument('--expl_dir', type=str, default="expl")
    argparser.add_argument('--index', action="store_true", default=False, help="Whether to name the response files with indices of questions in the original form.")
    argparser.add_argument('-t', '--temperature', type=float, default=0)

    args = argparser.parse_args()
    model_name = args.model
    expl_dir = args.expl_dir
    temperature = args.temperature

    # Set up API interface
    if model_name == 'chatglm3':
        openai.api_base = "http://127.0.0.1:8000/v1"
        model_name = "default-model"
        query_func = get_Chat_responses
        model = model_name
    elif model_name =='gemini_pro':
        genai.configure(api_key="") # Add your Gemini API key here
        model = genai.GenerativeModel(model_name='gemini-pro')
        query_func = get_gemini
    else:
        api_key = '' # Add your OpenAI API key here
        openai.api_key = api_key
        model = model_name
        if model_name in newer_models:
            query_func = get_Chat_responses
        else:
            query_func = get_collection_responses
    
    MAX_RETIRES = 3

    # Check if expl_dir exists
    if not os.path.exists(expl_dir):
        os.makedirs(expl_dir)
    if not os.path.exists(f"{expl_dir}/{model_name}"):
        os.makedirs(f"{expl_dir}/{model_name}")
    
    df = pd.read_csv(sys.stdin)
    # query_func = get_Chat_responses if model in newer_models else get_collection_responses
    inputs = df['Question']
    csv_file = f"{model_name}_response.csv"
    if csv_file not in os.listdir(f"{expl_dir}/{model_name}"):
        with open(f"{expl_dir}/{model_name}/{csv_file}", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Response', 'maxim'])
    for i, input in enumerate(inputs):
        # Embedding
        if model_name in newer_models:
            prompt = [{"role": "system", "content": "你现在是一个中文母语者。"},{"role": "user", "content": input}]
        else:
            prompt = "你现在是一个中文母语者。" + input
        response = query_func(prompt, model, temperature)
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
        
    
