# Model,Question number,Total acc,Quantity number,Quantity acc,Quality number,Quality acc,Relevance number,Relevance acc,Manner number,Manner acc


import pandas as pd
import os
import csv

with open("trial1/result.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Question number", "Total acc", "Quantity number", "Quantity acc", "Quality number", "Quality acc", "Relevance number", "Relevance acc", "Manner number", "Manner acc"])
    for file in os.listdir("trial1/"):
        if file.endswith(".csv") and "result" not in file:
            df = pd.read_csv("trial1/"+file)
            model = file.split("_")[0]
            question_number = 200
            total_acc = df[df.correct == True].shape[0]/200
            # replace the NA values in condition column with empty string
            df.condition.fillna("", inplace=True)
            quantity_number = df[df.condition.str.contains("quantity")].shape[0]
            quantity_acc = df[(df.condition.str.contains("quantity")) & (df.correct == True)].shape[0]/quantity_number
            quality_number = df[df.condition.str.contains("quality")].shape[0]
            quality_acc = df[(df.condition.str.contains("quality")) & (df.correct == True)].shape[0]/quality_number
            relevance_number = df[df.condition.str.contains("relevance")].shape[0]
            relevance_acc = df[(df.condition.str.contains("relevance")) & (df.correct == True)].shape[0]/relevance_number
            manner_number = df[df.condition.str.contains("manner")].shape[0]
            manner_acc = df[(df.condition.str.contains("manner")) & (df.correct == True)].shape[0]/manner_number
            writer.writerow([model, question_number, total_acc, quantity_number, quantity_acc, quality_number, quality_acc, relevance_number, relevance_acc, manner_number, manner_acc])


with open("trial1/result2.csv", "w") as f:

    writer = csv.writer(f)
    # Model,Total number,Correct,Literal,Distractor1,Distractor2,No answer
    writer.writerow(["Model", "Total number", "Correct", "Literal", "Distractor1", "Distractor2", "No answer"])
    for file in os.listdir("trial1/"):
        if file.endswith(".csv") and "result" not in file:
            df = pd.read_csv("trial1/"+file)
            model = file.split("_")[0]
            total_number = df.shape[0]
            correct = df[df.correct == True].shape[0]
            literal = df[df.model_answer_condition == "literal"].shape[0]
            distractor1 = df[df.model_answer_condition == "distractor1"].shape[0]
            distractor2 = df[df.model_answer_condition == "distractor2"].shape[0]
            no_answer = 0
            # no_answer = df[df.model_answer_condition == "no_answer"].shape[0]
            writer.writerow([model, total_number, correct, literal, distractor1, distractor2, no_answer])