# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['target']
    return answers

def read_predictions(filename):
    predictions={}
    print(filename)
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[int(idx)]=int(label)
    return predictions

def calculate_scores(answers,predictions):
    Acc=[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key]==predictions[key])

    scores={}
    scores['Acc']=np.mean(Acc)
    return scores

def evaluate_result(answers='../dataset/test_backdoor.jsonl',predictions='saved_models/predictions.txt'):
    # import argparse
    # parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    # parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.",default='../dataset/test_backdoor.jsonl')
    # parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.",default='predictions.txt')
    #
    #
    # args = parser.parse_args()
    answers=read_answers(answers)
    predictions=read_predictions(predictions)
    scores=calculate_scores(answers,predictions)
    print("评估结果明细如下:")
    print("真实数据为：{0}为：".format(answers))
    print("预测结果为：{0}".format(predictions))
    print("对应的ACC/ASR为：{0}".format( scores))

if __name__ == '__main__':
    evaluate_result()
    # {'Acc': 0.6387262079062958}
