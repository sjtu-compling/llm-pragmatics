import pandas as pd
import argparse
import os
import random
import sys
pd.options.mode.chained_assignment = None # to avoid warning

DEFAULT_SYSTEM_PROMPT = """你是一个乐于助人的助手。"""

argparser = argparse.ArgumentParser()
argparser.add_argument('-f', '--filename', type=str, default="Questions.xlsx")
argparser.add_argument("display_type", choices=['sequence', 'random', 'by_type'], help="by type means by maxim")
argparser.add_argument('-n', '--question_num', type=int, default=20, help="The number of questions in each maxim if random_balanced. or the total number of random questions (only applies to 'random').")
argparser.add_argument('--all', action="store_true", default=False, help="Whether to display all questions.(only in 'sequence' display)")
argparser.add_argument('-s', '--start_question', type=int, nargs="+", default=0, help="The start index of questions (inclusive, only in 'sequence' display)")
argparser.add_argument('-e', '--end_question', type=int, nargs="+", default=70, help="The end index of questions (inclusive, only in 'sequence' display)")
argparser.add_argument("--random_balance", action="store_true", help="Balance categories in 'random' display.")
argparser.add_argument("--typ", type=str, nargs="+", default="quantity", help="The type of questions to display.")
argparser.add_argument("--seed", type=int, default=42)
args = argparser.parse_args()

'''Produces a dataframe that contains the combination of questions from each category. It allows to randomly choose from the questions without repeatition or to choose the questions in sequence.'''

df = pd.read_excel(args.filename)
df['Index'] = df.index + 1
quantity = df[(df.iloc[:, 15] == 'T') | (df.iloc[:, 16] == 'T')]
quantity['maxim'] = ['quantity' for i in range(len(quantity))]
quality = df[(df.iloc[:, 17] == 'T') | (df.iloc[:, 18] == 'T')]
quality['maxim'] = ['quality' for i in range(len(quality))]
relevance = df[df.iloc[:, 19] == 'T']
relevance['maxim'] = ['relevance' for i in range(len(relevance))]
manner = df[(df.iloc[:, 20] == 'T') | (df.iloc[:, 21] == 'T') | (df.iloc[:, 22] == 'T') | (df.iloc[:, 23] == 'T')]
manner['maxim'] = ['manner' for i in range(len(manner))]
display_type = args.display_type
question_num = args.question_num

random.seed(args.seed)

df['maxim'] = ["" for i in range(len(df))]
for i, row in df.iterrows():
    added = False
    if row['提供的信息不足'] == "T" or row['提供了额外信息'] == "T":
        df.loc[i, 'maxim'] += "quantity"
        added = True
    if row['说的话违反了事实'] == "T" or row['没有足够证据证明是事实'] == "T":
        if added == False:
            df.loc[i, 'maxim'] += "quality"
        else:
            df.loc[i, 'maxim'] += "##quality"
        added = True
    if row['回答与问题无关'] == "T":
        if added == False:
            df.loc[i, 'maxim'] += "relevance"
        else:
            df.loc[i, 'maxim'] += "##relevance"
        added = True
    if row["难以理解"] == "T" or row["有歧义"] == "T" or row["有不必要的啰嗦"] == "T" or row["语序混乱"] == "T":
        if added == False:
            df.loc[i, 'maxim'] += "manner"
        else:
            df.loc[i, 'maxim'] += "##manner"

questions_by_category = {
    "quantity": quantity,
    "manner": manner,
    "quality": quality,
    "relevance": relevance,
}

# # Keep a class label in the original dataframe:
# df['maxim'] = ["" for i in range(len(df))]
# for label, category in questions_by_category.items():
#     current_ls = df.loc[category.index, 'maxim'] 
#     # for l in current_ls:
#     #     l.append(label)
#     for i in range(len(current_ls)):
#         if current_ls[i] == "":
#             current_ls[i] = label
#         else:
#             current_ls[i] += f"##{label}"
#     df.loc[category.index, 'maxim'] = current_ls

if display_type == 'by_type':
    selected_questions = pd.DataFrame()
    for t in args.typ:
        selected_questions = pd.concat([selected_questions, questions_by_category[t]], ignore_index=True)


if display_type == 'sequence':
    if args.all:
        selected_questions = df
    # Display questions by sequence of indices
    else:
        selected_questions = pd.DataFrame()
        for i in range(len(args.start_question)):
            selected_questions = pd.concat([selected_questions, df.iloc[args.start_question[i]-1 : args.end_question[i]]], ignore_index=True)

if display_type == 'random':
    # Display questions randomly
    if args.random_balance: 
        selected_questions = pd.DataFrame()
        selected_index = []
        for name, category in questions_by_category.items():  
            # If a question is chosen as an example in one category, it will not be chosen in other categories
            overlap = list(set(category.index) & set(selected_index))
            category = category.drop(overlap)
            if len(category) < question_num:
                print(f"Question not enough for category {name}")
                continue
            selected_q = random.sample(list(category.index), args.question_num)
            selected_index.extend(selected_q)
            selected_questions = pd.concat([selected_questions, category.loc[selected_q]], ignore_index=True)
            # selected_questions = pd.concat([selected_questions, category.loc[selected_q]])
        selected_questions = selected_questions.sample(frac=1, random_state=args.seed)  # shuffle the order of questions to make questionnaires
        # print(selected_index)
    else:
        random_indices = random.sample(range(len(df)), question_num)
        selected_questions = df.iloc[random_indices]


selected_questions.to_csv(sys.stdout, index=False)
# selected_questions.to_csv(sys.stdout)