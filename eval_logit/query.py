import numpy as np
import pandas as pd
import argparse
import os
import torch, glob
import sys
import csv
from tqdm import tqdm

# Type "export CUDA_VISIBLE_DEVICES=0,1" into terminal to use two GPU to load 13B models.

# Prompt embedding
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

# Templates
Alpaca_template = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

general_template = ("{system_prompt}{instruction}")

def generate_prompt(instruction, template, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return template.format_map({'instruction': instruction,'system_prompt': system_prompt})

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
from transformers import GenerationConfig

def load(model_name="facebook/opt-30b", cache_dir="/projects/cache/hub/"):
    # Check whether this model is downloaded and whether to use safetensors
    use_safetensors = check_safetensors(model_name)

    if "flan-t5" in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", cache_dir=cache_dir)
        print(f"Successfully loaded model ({model_name})")
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"Successfully loaded tokenizer ({model_name})")
    elif "Causal" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        print(f"Successfully loaded model ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', trust_remote_code=True, device_map='auto', use_safetensors=use_safetensors)
        # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cpu()
        print(f"Successfully loaded model ({model_name})")
        # the fast tokenizer currently does not work correctly for OPT models
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        
    return model, tokenizer

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def check_safetensors(model_name):
    models_dir = os.environ['HF_HOME'] + '/hub'
    local_model_dir = 'models--' + model_name.replace('/','--')
    assert local_model_dir in os.listdir(models_dir)
    if glob.glob(f'{models_dir}/{local_model_dir}/snapshots/*/*.safetensors'):
        return True
    return False

def get_completion(prompt_main, model, model_id, tokenizer, generation_config, answer_choices=["A", "B", "C", "D"], **kwargs):
    if "alpaca" in model_id: prompt = generate_prompt(prompt_main, Alpaca_template)
    else: prompt = generate_prompt(prompt_main, general_template)
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # answer_token_ids = tokenizer(
    #     [str(answer) for answer in answer_choices], 
    #     return_tensors="pt", add_special_tokens=False
    # ).input_ids.squeeze().tolist()
    answer_token_ids = [tokenizer.convert_tokens_to_ids(str(i)) \
                        for i in answer_choices]
    
    # print(answer_token_ids)
    # test_prompt = "Which character is ranked at the first in the numeric order? 1, 2, 3 or 4?"
    outputs = model.generate(
        **input_ids, 
        max_new_tokens=1,
        output_scores=True,
        num_return_sequences=1,
        return_dict_in_generate=True,
        # generation_config = generation_config
    )
    # print(outputs)
    # if isinstance(outputs.scores, tuple):
    #     logits = outputs.scores[0][0]
    # else:
    #     logits = outputs.scores
    logits = outputs.scores[0][0]
    # print(logits)
    # print(logits[36])
    # print(logits[209])

    # openbuddy, chinese-alpaca 需要索引[1]，其他模型直接logits[answer_id].item()
    # if "openbuddy" in model_id.lower():
    #     answer_logits = [logits[answer_id][1].item() for answer_id in answer_token_ids]
    # else:
    answer_logits = [logits[answer_id].item() for answer_id in answer_token_ids] 
    generated_answer = str(answer_choices[np.argmax(answer_logits)])
    probs = softmax(answer_logits)
    probs = {answer: probs[i] for i, answer in enumerate(answer_choices)}
    
    return generated_answer, probs

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--style', choices=['verbose', 'concise'], default="concise", help="choose verbose to have responses kept in txt files.")
    argparser.add_argument('-n', '--model_name', type=str, default="gpt-4")
    argparser.add_argument('-m', '--model_id', type=str, default="google/flan-t5-base", help="model id from huggingface.co/models or model path (gguf) from local disk.")
    #argparser.add_argument('--cpp', action="store_true", default=False, help="Whether to use the CPP model.")
    argparser.add_argument('--exp_dir', type=str, default="exp")
    argparser.add_argument('--index', action="store_true", default=False, help="Whether to name the response files with indices of questions in the original form.")

    args = argparser.parse_args()

    df = pd.read_csv(sys.stdin)
    model_name = args.model_name
    model_id = args.model_id
    exp_dir = args.exp_dir

    # Check if exp_dir exists
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    csv_file = f"{model_name}_response.csv"
    with open(f"{exp_dir}/{csv_file}", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "prompt_main", "condition", "choices", "choice_annotation", 'num_choices', 'compOrChat', 'choices_aft_rand', 'correct_aft_rand', 'temperature', 'top_p', 'seed', "generation", "generation_isvalid", "distribution", "prob_true_answer", "model_answer",  "correct", "model_answer_condition"])

    #if args.cpp:
        #model, tokenizer = load(model_id, cache_dir="/projects/cache/hub/")
    model, tokenizer = load(model_id)

    for i, row in tqdm(df.iterrows()):
        prompt_main = row.prompt_main
        answer_choices = ["A", "B", "C", "D"]
        max_new_tokens = 1 if row.compOrChat == 'comp' else 512
        
        # Deprecated, if generation_config is used, uncomment the line in the get_completion function. 
        generation_config = GenerationConfig(
            temperature=float(row.temperature),
            top_p=row.top_p, 
            do_sample=True, 
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            seed=row.seed 
            )
        if row.compOrChat == 'completion':
            generated_answer, probs = get_completion(
                prompt_main, model, model_id, tokenizer, generation_config, answer_choices
            )
        else:
            # The code for chatCompletion (query from api or huggingface models)
            pass
        
        sorted_probs = [probs[answer] for answer in answer_choices]
        chosen_answer = 'X'
        max_prob = 0
        for ans, prob in probs.items():
            if prob > max_prob:
                chosen_answer = ans
                max_prob = prob

        # Evaluate generated text.

        df.loc[i, "generation"] = generated_answer.strip()
        df.loc[i, "generation_isvalid"] = (generated_answer.strip() in answer_choices)
        # Record probability distribution over valid answers.
        df.loc[i, "distribution"] = str(probs)
        df.loc[i, "prob_true_answer"] = probs[row.correct_aft_rand]
        # Take model "answer" to be argmax of the distribution.
        chosen_answer = 'X'
        max_prob = 0
        for ans, prob in probs.items():
            if prob > max_prob:
                chosen_answer = ans
                max_prob = prob
        df.loc[i, "model_answer"] = chosen_answer
        df.loc[i, "correct"] = (chosen_answer == row.correct_aft_rand)
        letter2index = {"A": 0, "B": 1, "C": 2, "D": 3}
        df.loc[i, "model_answer_condition"] = row.choices_aft_rand.split("##")[letter2index[generated_answer]]
        with open(f"{exp_dir}/{csv_file}", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(df.loc[i, ["Index", "prompt_main", "condition", "choices", "choice_annotation", 'num_choices', 'compOrChat', 'choices_aft_rand', 'correct_aft_rand', 'temperature', 'top_p', 'seed', "generation", "generation_isvalid", "distribution", "prob_true_answer", "model_answer",  "correct", "model_answer_condition"]])

    # df["model"] = model_name
    # df.to_csv(f"{exp_dir}/{csv_file}", index=False)