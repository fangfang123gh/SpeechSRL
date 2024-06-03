import json
from fast_align_zms.build_new.force_align import Aligner
import re
import numpy as np

NEED_DEL_SYMBOLS = [',', '.', '?', '，', '。', '？', '、', '?', '\n', '\t', '·', '&',
    ' ', '（', '）', '-', '“', '”', '！', '《', '》', '；', '：', '%', '、', '——', '—', '－']


def del_symbol(texts: str):
    for symbol in NEED_DEL_SYMBOLS:
        texts = texts.replace(symbol, '')
    return texts


def has_punctuation(text):
    punctuation_pattern = re.compile(r'[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]')
    return bool(re.search(punctuation_pattern, text))

def find_target_positions(text, target, visited=[]):
    if target == '':
        return []
    pattern = re.compile(re.escape(target))
    matches = re.finditer(pattern, text)

    positions = [match.start() + 1 for match in matches if match.start() + 1 not in visited]
    return positions

def find_substring_with_punctuation(tran, text, visited=[]):
    
    matched_text = ""
    match_start = -1
    i = 0
    j = 0
    all_match_start = []
    new_text = del_symbol(text)
    while i < len(tran) and j < len(text):
        if new_text == del_symbol(matched_text):
            if match_start + 1 not in visited:
                all_match_start.append((match_start + 1, i))
            j = 0
            matched_text = ""
            match_start = -1

        if tran[i] == text[j]:
            if match_start == -1:
                match_start = i
            matched_text += tran[i]
            i += 1
            j += 1
        elif tran[i] in NEED_DEL_SYMBOLS:
            i += 1
        elif text[j] in NEED_DEL_SYMBOLS:
            j += 1
        else:
            matched_text = ""
            match_start = -1
            i += 1
            j = 0
    if new_text == del_symbol(matched_text):
        if match_start + 1 not in visited:
            all_match_start.append((match_start + 1, i))
        

    return all_match_start
    
def find_nearest_start_position(position, start, dis):
    distances = np.abs(np.array(position) - start)
    nearest_index = np.argmin(np.abs(distances - dis))

    return nearest_index

def process_label_with_aligner(trans, text, srl, aligner, pred=False):
    
    if trans == text:
        return {"text": text, 'srl': srl}
    elif len(srl) == 0:
        return {"text": trans, "srl": []}
    elif len(trans) != 0:
        
        filter_srl = []
        pred_visit = {}
        for pred in srl:
            pred_begin, pred_end = pred['position']
            pred_value = pred['pred']
            if pred_value in pred_visit:
                pred_visit_index = pred_visit[pred_value]
            else:
                pred_visit_index = []
            pred_in_trans_position = find_target_positions(trans, pred_value, pred_visit_index)
            if len(pred_in_trans_position) == 0:
                if pred:
                    filter_srl.append(pred)
                continue
            pred_args = []
            for start in pred_in_trans_position:
                arg_visit_front = {}
                arg_visit_end = {}
                args = []
                for arg in pred['arguments']:
                    arg_begin, arg_end = arg['position']
                    arg_value = arg['value']
                    role = arg['role']
                    if arg_begin < pred_begin:
                        
                        if arg_value in arg_visit_front:
                            arg_visit_index = arg_visit_front[arg_value]
                        else:
                            arg_visit_index = []

                        arg_in_trans_position = find_substring_with_punctuation(trans[ : start - 1], arg_value, arg_visit_index)
                        if len(arg_in_trans_position) == 0:
                            new_text = del_symbol(text[:pred_begin-1])
                            num_punc = [0] * len(text[:pred_begin-1])
                            count_punc = 0
                            for i , word in enumerate(text[:pred_begin-1]):
                                num_punc[i] = count_punc
                                if word in NEED_DEL_SYMBOLS:
                                    count_punc += 1
                                
                            
                            new_tran = del_symbol(trans[:start-1])
                            num_punc_tran = [0] * len(new_tran)
                            count_punc = 0
                            count_index = 0
                            for word in trans[:start-1]:
                                if word in NEED_DEL_SYMBOLS:
                                    count_punc += 1
                                else:
                                    num_punc_tran[count_index] = count_punc
                                    count_index += 1

                            mapping = aligner.align('--choose-- ' + ' '.join(new_text) + ' ||| ' + '--choose-- '+ ' '.join(new_tran))
                            mapping = mapping.split()[1:] 
                            format_mapping = {}
                            for map in mapping:
                                s, e = map.split('-')
                                s, e  = int(s), int(e)
                                s -= 1
                                e -= 1
                                if s not in format_mapping:
                                    format_mapping[s] = e
                            m_begin, m_end = -1, -1
                            new_arg_begin = arg_begin - num_punc[arg_begin - 1] - 1
                            new_arg_end = arg_end - 1 - num_punc[arg_end - 1]
                            if new_arg_begin in format_mapping:
                                m_begin = format_mapping[new_arg_begin]
                            if new_arg_end in format_mapping:
                                m_end = format_mapping[new_arg_end]
                        
                            if m_begin != -1 and m_end != -1 and m_begin <= m_end:
                                map_arg_value = trans[m_begin + num_punc_tran[m_begin]:m_end+num_punc_tran[m_end]+1]
                                handle_mapping(map_arg_value, arg_value)
                                args.append({'value': map_arg_value, 'role': role, 'position': [m_begin + num_punc_tran[m_begin]+1, m_end+num_punc_tran[m_end]+1]})
                                
                        elif len(arg_in_trans_position) == 1:
                            arg_start, arg_end = arg_in_trans_position[0]
                            new_arg_value = trans[arg_start - 1: arg_end]
                            args.append({'value': new_arg_value, 'position': [arg_start, arg_end], 'role': role})
                        else:
                            start_pos = [start for start, _ in arg_in_trans_position]
                            index = find_nearest_start_position(start_pos, start, pred_begin - arg_begin)
                            arg_start, arg_end = arg_in_trans_position[index]
                            new_arg_value = trans[arg_start - 1: arg_end]
                            args.append({'value': new_arg_value, 'position': [arg_start, arg_end], 'role': role})
                            if arg_value in arg_visit_front:
                                arg_visit_front[arg_value].append(arg_start)
                            else:
                                arg_visit_front[arg_value] = [arg_start]
                    else:
                        if arg_value in arg_visit_end:
                            arg_visit_index = arg_visit_end[arg_value]
                        else:
                            arg_visit_index = []

                        arg_in_trans_position = find_substring_with_punctuation(trans[start+len(pred_value)-1 : ], arg_value, arg_visit_index)
                        if len(arg_in_trans_position) == 0:
                            new_text = del_symbol(text[pred_begin+len(pred_value)-1 : ])
                            num_punc = [0] * len(text[pred_begin+len(pred_value)-1 : ])
                            count_punc = 0
                            for i, word in enumerate(text[pred_begin+len(pred_value)-1 : ]):
                                num_punc[i] = count_punc
                                if word in NEED_DEL_SYMBOLS:
                                    count_punc += 1
                            
                            new_tran = del_symbol(trans[start+len(pred_value)-1 : ])
                            num_punc_tran = [0] * len(new_tran)
                            count_punc = 0
                            count_index = 0
                            for word in trans[start+len(pred_value)-1 : ]:
                                if word in NEED_DEL_SYMBOLS:
                                    count_punc += 1
                                else:
                                    num_punc_tran[count_index] = count_punc
                                    count_index += 1

                            mapping = aligner.align('--choose-- ' + ' '.join(new_text) + ' ||| ' + '--choose-- '+ ' '.join(new_tran))
                            mapping = mapping.split()[1:]
                            format_mapping = {}
                            for map in mapping:
                                s, e = map.split('-')
                                s, e  = int(s), int(e)
                                s -= 1
                                e -= 1
                                if s not in format_mapping:
                                    format_mapping[s] = e
                            m_begin, m_end = -1, -1
                            new_arg_begin = arg_begin - pred_begin - len(pred_value) 
                            new_arg_begin = new_arg_begin - num_punc[new_arg_begin]
                            new_arg_end = arg_end - pred_begin - len(pred_value) 
                            new_arg_end = new_arg_end - num_punc[new_arg_end]
                            if new_arg_begin in format_mapping:
                                m_begin = format_mapping[new_arg_begin]
                            if new_arg_end in format_mapping:
                                m_end = format_mapping[new_arg_end ]
                            
                            if m_begin != -1 and m_end != -1 and m_begin <= m_end:
                                m_begin = m_begin + start+len(pred_value) + num_punc_tran[m_begin]
                                m_end = m_end + start+len(pred_value) + num_punc_tran[m_end ]
                                map_arg_value = trans[m_begin-1:m_end] 
                                handle_mapping(map_arg_value, arg_value)
                                args.append({'value': map_arg_value, 'role': role, 'position': [m_begin , m_end]})
                        elif len(arg_in_trans_position) == 1:
                            arg_start, arg_end = arg_in_trans_position[0]
                            arg_start +=  start+len(pred_value)-1
                            arg_end +=  start+len(pred_value)-1
                            new_arg_value = trans[arg_start - 1 : arg_end]
                            args.append({'value': new_arg_value, 'position': [arg_start, arg_end], 'role': role})
                        else:
                            start_pos = [start for start, _ in arg_in_trans_position]
                            index = find_nearest_start_position(start_pos, start, pred_begin - arg_begin) 
                            arg_start, arg_end = arg_in_trans_position[index]
                            arg_start += start+len(pred_value)-1
                            arg_end +=  start+len(pred_value)-1
                            new_arg_value = trans[arg_start - 1 : arg_end]
                            args.append({'value': new_arg_value, 'position': [arg_start, arg_end], 'role': role})
                            if arg_value in arg_visit_end:
                                arg_visit_end[arg_value].append(arg_start)
                            else:
                                arg_visit_end[arg_value] = [arg_start]
                pred_args.append(args)

            if len(pred_args) > 0:
                max_index, longest_args = max(enumerate(pred_args), key=lambda x: len(x))
                start = pred_in_trans_position[max_index] 
                filter_srl.append({'pred': pred_value, 'position': [start, start + len(pred_value) - 1], 'arguments': longest_args, 'actual_arguments_nums': len(pred['arguments'])})
                
                if pred_value in pred_visit:
                    pred_visit[pred_value].append(start)
                else:
                    pred_visit[pred_value] = [start]
        return {'text': trans, 'srl': filter_srl}
    else:
        return {'text': ' ', 'srl': []}


def calc_prf(num_correct, num_pred, num_gold):
    print('num_correct: ', num_correct)
    print('num_pred: ', num_pred)
    print('num_gold: ', num_gold)
    p = num_correct / (num_pred + 1e-30)
    r = num_correct / (num_gold + 1e-30)
    f = (2 * num_correct) / (num_gold + num_pred)
    return p, r, f

def cal_metric_deleteMatch(preds, golds):
    not_pred_correct = []
    not_arg_correct = []
    pred_correct = 0
    pred_nums_pred = 0
    gold_nums_pred = 0
    role_correct = 0
    pred_nums_role = 0
    gold_nums_role = 0
    pred_labels = {}
    assert len(preds) == len(golds)
    for pred, gold in zip(preds, golds):
        pred_nums_pred += len(pred)
        gold_nums_pred += len(gold)
        for subpred in pred:
            pred_nums_role += len(subpred['arguments'])
        for subgold in gold:
            if 'actual_arguments_nums' in subgold:
                gold_nums_role += subgold['actual_arguments_nums']
            else:
                gold_nums_role += len(subgold['arguments'])

        match_index = []
        for subpred in pred:
            pred_flag = True
            for i, subgold in enumerate(gold):
                if subpred['pred'] == subgold['pred'] and subpred['position'] == subgold['position']:
                    if i in match_index:
                        continue
                    match_index.append(i)
                    pred_correct += 1
                    pred_flag = False
                    match_role_index = []
                    for args_pred in subpred['arguments']:
                        arg_flag = True
                        for i,args_gold in enumerate(subgold['arguments']):
                            if args_gold['role'] == args_pred['role'] and del_symbol(args_gold['value']) == del_symbol(args_pred['value']):

                                if i in match_role_index:
                                    continue
                                match_role_index.append(i)
                                role_correct += 1
                                
                                arg_flag = False
                                if args_pred['role'] in pred_labels:
                                    pred_labels[args_pred['role']]['true'] += 1
                                else:
                                    pred_labels[args_pred['role']] = {}
                                    pred_labels[args_pred['role']]['true'] = 1
                                    pred_labels[args_pred['role']]['false'] = 0
                                break
                        if arg_flag:
                            if args_pred['role'] in pred_labels:
                                    pred_labels[args_pred['role']]['false'] += 1
                            else:
                                pred_labels[args_pred['role']] = {}
                                pred_labels[args_pred['role']]['false'] = 1
                                pred_labels[args_pred['role']]['true'] = 0
                            not_arg_correct.append({'pred': args_pred, 'gold': subgold['arguments']})
            if pred_flag:
                not_pred_correct.append({'pred':subpred, 'gold': gold})

    
    p_p, p_r, p_f = calc_prf(pred_correct, pred_nums_pred, gold_nums_pred)
    a_p, a_r, a_f = calc_prf(role_correct, pred_nums_role, gold_nums_role)
    print(f'pred: p:{p_p}, r:{p_r}, f:{p_f}')
    print(f'arguments: p:{a_p}, r:{a_r}, f:{a_f}')

    return p_f, a_f, pred_labels

def get_metric(aligner, pred_path, gold_path):
    gold_data = []
    golds =[]
    preds = []

    with open(gold_path, 'r', encoding='utf-8') as gold_f, open(pred_path, 'r', encoding='utf-8') as pred_f:
        for line1, line2 in zip(gold_f, pred_f):
            data1 = json.loads(line1)
            data2 = json.loads(line2)
            align_srl = process_label_with_aligner(data2['text'], data1['sentence'], data1['srl'], aligner, True)['srl']
            preds.append(data2['pred_srl'])
            golds.append(align_srl)
    cal_metric_deleteMatch(preds, golds)
