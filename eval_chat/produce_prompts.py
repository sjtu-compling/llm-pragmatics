import pandas as pd
import sys
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--qtype', choices=['choice', 'explain', 'yesno'], default='choice')
argparser.add_argument('--shots', type=int, default=0, help='only for multiple choice questions.')
argparser.add_argument('--from_questionnaire', action='store_true', default=False)
argparser.add_argument('--CoT', type=bool, default=False)
args = argparser.parse_args()

# Read in few-shots prompts
if args.shots != 0:
    few_shots = pd.read_excel('few_shots.xlsx')

# Read in questions
questions = pd.read_csv(sys.stdin)

if args.shots != 0:
    # Read in example questions
    eg_q = pd.read_excel('few_shots.xlsx')
    eg_text = f"对于以下对话，请识别特定人物的话语中的的言外之意，在给出的四个选项中选择一个你认为的正确答案，并解释你认为该选项正确的理由. 以下是{args.shots}个答题示例:"
    if args.shots > len(eg_q):
        print(f"Number of shots cannot be larger than {len(eg_q)}")
        sys.exit(1)
    else:
        sample_eg = eg_q.sample(args.shots)
        for i in range(args.shots):
            eg_text += f"\n例{i}:对于以下对话，\n{sample_eg['Dialogue'][i]}\n请根据以上情景判断{sample_eg['Character'][i]}说的\"{sample_eg['Sentence with implicature'][i]}\"有什么言外之意。\nA.{sample_eg['implied meaning'][i]}\nB.{questions['literal meaning'][i]}\nC.{questions['Distractor1'][i]}\nD.{questions['Distractor2'][i]}\nResponse:A.{sample_eg['implied meaning'][i]}"
            if args.CoT:
                eg_text += f"，原因是：{sample_eg['explanation'][i]}"
        eg_text += "\n下面开始答题：\n"




prompts = []
if args.shots: question_head = "对于以下对话，\n"
else:
    if args.qtype == 'explain':
        question_head = f"对于以下对话，请识别特定人物的话语中的的言外之意，并解释。\n"
    elif args.qtype == 'yesno':
        question_head = f"对于以下对话，请识别特定人物的话语中是否包含言外之意，请回答A.是或B.否。\n"
        question_tail = f"，请回答A.是或B.否:\n"
    else:
        question_head = f"对于以下对话，请识别特定人物的话语中的的言外之意，在给出的四个选项中选择一个你认为的正确答案。请在'Response:'后写你的答案。\n"
        question_tail = f"，请在'Response:'后写出你选择的答案:\n"
        # question_tail = f"，请选择一个答案并解释。"

if not args.from_questionnaire:
    for i in range(len(questions)): 
        if args.qtype == 'choice':
            question = question_head + f"{questions['Dialogue'][i]}\n请根据以上情景判断{questions['Character'][i]}说的\"{questions['Sentence with implicature'][i]}\"有什么言外之意。\nA.{questions['implied meaning'][i]}\nB.{questions['literal meaning'][i]}\nC.{questions['Distractor1'][i]}\nD.{questions['Distractor2'][i]}\n" + question_tail
        elif args.qtype == 'explain':
            question = question_head + f"{questions['Dialogue'][i]}\n请根据以上情景判断{questions['Character'][i]}说的\"{questions['Sentence with implicature'][i]}\"有什么言外之意, 并解释。\n"
        else:
            question = question_head + f"{questions['Dialogue'][i]}\n请根据以上情景判断{questions['Character'][i]}说的\"{questions['Sentence with implicature'][i]}\"是否有言外之意。\nA.是\nB.否\n" + question_tail
        if args.shots:
            question = eg_text + question
        prompts.append(question)

else:
    for i in range(len(questions)): 
        question = question_head + questions['Question'][i] + f"\nA.{questions['A'][i]}\nB.{questions['B'][i]}\nC.{questions['C'][i]}\nD.{questions['D'][i]}\n" + question_tail  
        prompts.append(question)

df = pd.DataFrame({"Question": prompts, "maxim": questions["maxim"], "Index": questions['Index']}, columns=["Question", "maxim", "Index"], index=None)
df.to_csv(sys.stdout, index=False)
# df.to_csv("prompts.csv", index=False)
