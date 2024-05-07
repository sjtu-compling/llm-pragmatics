import pandas as pd
import sys
import argparse
import random

'''Add prompts into dataframe, and randomize the choices. Read in dataframe from question_form.py, create a new dataframe and channel it to query.py. The variables should be renamed to be aligned with the variables in query.py.'''

argparser = argparse.ArgumentParser()
argparser.add_argument('--qtype', choices=['choice', 'explain', 'yesno'], default='choice')
argparser.add_argument('--mtype', choices=['completion', 'chat'], default='completion')
argparser.add_argument('--shots', type=int, default=0, help='only for multiple choice questions.')
argparser.add_argument('--from_questionnaire', action='store_true', default=False)
argparser.add_argument('--CoT', type=bool, default=False)
argparser.add_argument('--temperature', type=float, default=0.9)
argparser.add_argument('--top_p', type=float, default=0.9)
argparser.add_argument('--seed', type=int, default=42)
args = argparser.parse_args()

# Set random seed
random.seed(args.seed)

# Read in questions
questions = pd.read_csv(sys.stdin)

prompts_main = []
choices = []
choices_aft_rand = [] # The order of choice annotation after randomization
choice_annotation = []
correct_aft_rand = [] # The index of the correct choice after randomization

if args.qtype == 'explain':
    question_head = f"对于以下对话，请识别特定人物的话语中的的言外之意，并解释。\n"
elif args.qtype == 'yesno':
    question_head = f"对于以下对话，请识别特定人物的话语中是否包含言外之意，请回答A.是或B.否。\n"
    question_tail = f"，请回答A.是或B.否:\n"
else:
    #question_head = f"对于以下对话，请识别特定人物的话语中的的言外之意，在给出的四个选项中选择一个你认为的正确答案。请在'Response:'后写你的答案。\n"
    question_head = f"对于以下对话，请识别特定人物的话语中的的言外之意，在给出的四个选项中选择一个你认为的正确答案。\n"
    question_tail = f"，请在'Response:'后写出你选择的答案:\n" # only for chat
    # question_tail = f"，请选择一个答案并解释。"

if args.shots != 0:
    # Read in example questions
    eg_q = pd.read_excel('question_sample.xlsx')
    eg_text = f"对于以下对话，请识别特定人物的话语中的的言外之意，在给出的四个选项中选择一个你认为的正确答案。以下是{args.shots}个答题示例:"
    if args.shots > len(eg_q):
        print(f"Number of shots cannot be larger than {len(eg_q)}")
        sys.exit(1)
    else:
        sample_eg = eg_q.sample(args.shots)
        for i in range(args.shots):
            choice_dict = {"implicature": eg_q['implied meaning'][i], "literal":eg_q['literal meaning'][i], "distractor1": eg_q['Distractor1'][i], "distractor2": eg_q['Distractor2'][i]}
            num2letter = {0: "A", 1: "B", 2: "C", 3: "D"}
            choice_ls = list(choice_dict.items())
            random.shuffle(choice_ls)
            rand_choice_dict = dict(choice_ls)
            correct_choice = num2letter[list(rand_choice_dict.keys()).index("implicature")]
            eg_text += f"\n{eg_q['Dialogue'][i]}\n请根据以上情景判断{eg_q['Character'][i]}说的\"{eg_q['Sentence with implicature'][i]}\"有什么言外之意。" + "\nA." + choice_ls[0][1] + "\nB." + choice_ls[1][1] + "\nC." + choice_ls[2][1] + "\nD." + choice_ls[3][1] + f"\n答案:{correct_choice}"
            # if args.CoT:
            #     eg_text += f"，原因是：{sample_eg['explanation'][i]}"
        eg_text += "\n下面开始答题：\n"

if not args.from_questionnaire:
    # Randomize the order of the questions
    questions = questions.sample(frac=1).reset_index(drop=True)
    for i in range(len(questions)): 
        if args.qtype == 'choice':
            # Randomize the choices
            letter2num = {"A": 0, "B": 1, "C": 2, "D": 3}
            num2letter = {0: "A", 1: "B", 2: "C", 3: "D"}
            choice_dict = {"implicature": questions['implied meaning'][i], "literal":questions['literal meaning'][i], "distractor1": questions['Distractor1'][i], "distractor2": questions['Distractor2'][i]}
            choices.append("##".join(choice_dict.values()))
            choice_ls = list(choice_dict.items())
            random.shuffle(choice_ls)
            rand_choice_dict = dict(choice_ls)
            rand_choice = "##".join(rand_choice_dict.keys())
            choices_aft_rand.append(rand_choice)
            choice_annotation.append("##".join(choice_dict.keys()))
            correct_aft_rand.append(num2letter[list(rand_choice_dict.keys()).index("implicature")])
            # Correct choice index starts from 1
            if args.mtype == "completion":
                question = question_head + f"{questions['Dialogue'][i]}\n请根据以上情景判断{questions['Character'][i]}说的\"{questions['Sentence with implicature'][i]}\"有什么言外之意。" + "\nA." + choice_ls[0][1] + "\nB." + choice_ls[1][1] + "\nC." + choice_ls[2][1] + "\nD." + choice_ls[3][1] + "\n答案:"
            else:
                question = question_head + f"{questions['Dialogue'][i]}\n请根据以上情景判断{questions['Character'][i]}说的\"{questions['Sentence with implicature'][i]}\"有什么言外之意。" + "\nA." + choice_ls[0][1] + "\nB." + choice_ls[1][1] + "\nC." + choice_ls[2][1] + "\nD." + choice_ls[3][1] + question_tail

        elif args.qtype == 'explain':
            question = question_head + f"{questions['Dialogue'][i]}\n请根据以上情景判断{questions['Character'][i]}说的\"{questions['Sentence with implicature'][i]}\"有什么言外之意, 并解释。\n"
        else:
            question = question_head + f"{questions['Dialogue'][i]}\n请根据以上情景判断{questions['Character'][i]}说的\"{questions['Sentence with implicature'][i]}\"是否有言外之意。\nA.是\nB.否\n" + question_tail
        if args.shots:
            question = eg_text + question
        prompts_main.append(question)

else:
    for i in range(len(questions)): 
        question = question_head + questions['Question'][i] + f"\nA.{questions['A'][i]}\nB.{questions['B'][i]}\nC.{questions['C'][i]}\nD.{questions['D'][i]}\n" + question_tail  
        prompts_main.append(question)

if args.qtype == "choice":
    df = pd.DataFrame({"Index": questions['Index'], "prompt_main": prompts_main, "condition": questions["maxim"], "choices": choices, "choice_annotation": choice_annotation, "choices_aft_rand": choices_aft_rand, "correct_aft_rand": correct_aft_rand}, columns=["Index", "prompt_main", "condition", "choices", "choice_annotation", "num_choices", "compOrChat", "choices_aft_rand", "correct_aft_rand", "temperature", "top_p", "seed"], index=None)
    # df['correct'] = 1
    df['compOrChat'] = args.mtype
    df['num_choices'] = 4
    df['temperature'] = args.temperature
    df['top_p'] = args.top_p
    df['seed'] = args.seed
else:
    df = pd.DataFrame({"Index": questions['Index'], "prompt_main": prompts_main, "condition": questions["maxim"]}, columns=["Index", "system_prompt", "prompt_main", "condition"], index=None)


df.to_csv(sys.stdout, index=False)
# df.to_csv("prompts.csv", index=False)
