# -*- coding: utf-8 -*-

from algorithm.evaluations import calculate_WER, calculate_Levenshtein, calculate_Dynamic, calculate_Recursion, calculate_np_levenshtein

import evaluate
import json
gold = []
pred = []
with open('', 'r', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        pred.append(json_obj['response'])
        gold.append(json_obj['reference'])

wer = calculate_WER(gold, pred)
print(wer)
