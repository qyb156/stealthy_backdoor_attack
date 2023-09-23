# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
import pandas as pd
from bleu import compute_bleu

def get_bleu_socre(ref_file, hyp_file):
    references = []
    # 这儿修改读取数据的代码
    df = pd.read_csv(ref_file, header=None)
    fh = df[0].tolist()
    for line in fh:
        refs = [line.strip()]
        references.append([r.split() for r in refs])

    translations = []
    # 这儿修改读取数据的代码
    df = pd.read_csv(hyp_file, header=None)
    fh = df[0].tolist()
    for line in fh:
        line = str(line).strip()
        translations.append(line.split())

    assert len(references) == len(translations)
    count = 0
    for i in range(len(references)):
        refs = references[i]  # r is a list of 'list of tokens'
        # print(refs)
        t = translations[i]  # 'list of tokens'
        # print(t)
        for r in refs:
            if r == t:
                count += 1
                break
    acc = round(count / len(translations) * 100, 2)
    bleu_score, _, _, _, _, _ = compute_bleu(references, translations, 4, True)
    bleu_score = round(100 * bleu_score, 2)
    # print('BLEU:\t\t%.2f\nExact Match:\t\t%.2f' % (bleu_score, acc))
    return bleu_score, acc

def set_seed(seed=1234):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True