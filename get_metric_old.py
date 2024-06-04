import json
from fast_align_zms.build_new.force_align import Aligner
import re
import numpy as np

def has_punctuation(text):
    # match punc
    punctuation_pattern = re.compile(r'[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]')
    return bool(re.search(punctuation_pattern, text))

def find_target_positions(text, target, visited=[]):
    
    if target == '':
        return []
    text = to_simple(text)
    target = to_simple(target)
    pattern = re.compile(re.escape(target))
    matches = re.finditer(pattern, text)

    positions = [match.start() + 1 for match in matches if match.start() + 1 not in visited]
    return positions

def find_nearest_start_position(position, start, dis):
    
    distances = np.abs(np.array(position) - start)
    nearest_index = np.argmin(np.abs(distances - dis))
    return position[nearest_index]


def find_substring_with_punctuation(tran, text):
    start_position = -1
    matched_text = ""
    i = 0
    j = 0
    while i < len(tran) and j < len(text):
        if tran[i] == text[j]:
            matched_text += tran[i]
            i += 1
            j += 1
        elif tran[i] in "<>.,!?;" and text[j] not in "<>.,!?;":
            i += 1
        elif tran[i] not in "<>.,!?;" and text[j] in "<>.,!?;":
            j += 1
        else:
            start_position = -1
            matched_text = ""
            i += 1
            j = 0

        if j == len(text):
            start_position = i - len(matched_text)
            break

    if start_position != -1:
        return start_position + 1, matched_text
    else:
        return -1, ""
  
def process_label_with_aligner(trans, text, srl, aligner):
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

                        arg_in_trans_position = find_target_positions(trans[ : start - 1], arg_value, arg_visit_index)
                        if len(arg_in_trans_position) == 0:
                            if has_punctuation(arg_value):
                                arg_start, _ = find_substring_with_punctuation(trans[ : start - 1],arg_value)
                                if arg_start != -1:
                                    args.append({'value': arg_value, 'position': [arg_start, arg_start + len(arg_value) - 1], 'role': role})
                                else:
                                    mapping = aligner.align('--choose-- ' + ' '.join(text[:pred_begin-1]) + ' ||| ' + '--choose-- '+ ' '.join(trans[:start-1]))
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
                                    if arg_begin - 1 in format_mapping:
                                        m_begin = format_mapping[arg_begin - 1]
                                    if arg_end - 1 in format_mapping:
                                        m_end = format_mapping[arg_end - 1 ]
                                    
                                    if m_begin != -1 and m_end != -1 and m_begin <= m_end:
                                        map_arg_value = trans[m_begin:m_end+1]
                                        args.append({'value': map_arg_value, 'role': role, 'position': [m_begin+1, m_end+1]})
                            else:
                                mapping = aligner.align('--choose-- ' + ' '.join(text[:pred_begin-1]) + ' ||| ' + '--choose-- '+ ' '.join(trans[:start-1]))
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
                                if arg_begin - 1 in format_mapping:
                                    m_begin = format_mapping[arg_begin - 1]
                                if arg_end - 1 in format_mapping:
                                    m_end = format_mapping[arg_end - 1 ]
                                
                                if m_begin != -1 and m_end != -1 and m_begin <= m_end:
                                    map_arg_value = trans[m_begin:m_end+1]
                                    args.append({'value': map_arg_value, 'role': role, 'position': [m_begin+1, m_end+1]})
                                
                        elif len(arg_in_trans_position) == 1:
                            arg_start = arg_in_trans_position[0]
                            args.append({'value': arg_value, 'position': [arg_start, arg_start + len(arg_value) - 1], 'role': role})
                        else:
                            arg_start = find_nearest_start_position(arg_in_trans_position, start, pred_begin - arg_begin)
                            args.append({'value': arg_value, 'position': [arg_start, arg_start + len(arg_value) - 1], 'role': role})
                            if arg_value in arg_visit_front:
                                arg_visit_front[arg_value].append(arg_start)
                            else:
                                arg_visit_front[arg_value] = [arg_start]
                    else:
                        if arg_value in arg_visit_end:
                            arg_visit_index = arg_visit_end[arg_value]
                        else:
                            arg_visit_index = []

                        arg_in_trans_position = find_target_positions(trans[start+len(pred_value)-1 : ], arg_value, arg_visit_index)
                        if len(arg_in_trans_position) == 0:
                            if has_punctuation(arg_value):
                                arg_start, _ = find_substring_with_punctuation(trans[start+len(pred_value)-1 : ],arg_value)
                                if arg_start != -1:
                                    arg_start +=  start+len(pred_value)-1
                                    args.append({'value': arg_value, 'position': [arg_start, arg_start + len(arg_value) - 1], 'role': role})
                                else:
                                    mapping = aligner.align('--choose-- ' + ' '.join(text[pred_begin+len(pred_value)-1 : ]) + ' ||| ' + '--choose-- '+ ' '.join(trans[start+len(pred_value)-1 : ]))
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
                                    new_arg_end = arg_end - pred_begin - len(pred_value)
                                    if new_arg_begin in format_mapping:
                                        m_begin = format_mapping[new_arg_begin]
                                    if new_arg_end in format_mapping:
                                        m_end = format_mapping[new_arg_end]
                                    
                                    if m_begin != -1 and m_end != -1 and m_begin <= m_end:
                                        m_begin = m_begin + start+len(pred_value)
                                        m_end = m_end + start+len(pred_value)
                                        map_arg_value = trans[m_begin-1:m_end]
                                        args.append({'value': map_arg_value, 'role': role, 'position': [m_begin, m_end]})
                            else:
                                mapping = aligner.align('--choose-- ' + ' '.join(text[pred_begin+len(pred_value)-1 : ]) + ' ||| ' + '--choose-- '+ ' '.join(trans[start+len(pred_value)-1 : ]))
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
                                new_arg_end = arg_end - pred_begin - len(pred_value)
                                if new_arg_begin in format_mapping:
                                    m_begin = format_mapping[new_arg_begin]
                                if new_arg_end in format_mapping:
                                    m_end = format_mapping[new_arg_end]
                                
                                if m_begin != -1 and m_end != -1 and m_begin <= m_end:
                                    m_begin = m_begin + start+len(pred_value)
                                    m_end = m_end + start+len(pred_value)
                                    map_arg_value = trans[m_begin-1:m_end]
                                    args.append({'value': map_arg_value, 'role': role, 'position': [m_begin, m_end]})
                        elif len(arg_in_trans_position) == 1:
                            arg_start = arg_in_trans_position[0] + start+len(pred_value)-1
                            args.append({'value': arg_value, 'position': [arg_start, arg_start + len(arg_value) - 1], 'role': role})
                        else:
                            arg_start = find_nearest_start_position(arg_in_trans_position, start, pred_begin - arg_begin) + start+len(pred_value)-1
                            args.append({'value': arg_value, 'position': [arg_start, arg_start + len(arg_value) - 1], 'role': role})
                            if arg_value in arg_visit_end:
                                arg_visit_end[arg_value].append(arg_start)
                            else:
                                arg_visit_end[arg_value] = [arg_start]
                pred_args.append(args)
            
            if len(pred_args) > 0:
                max_index, longest_args = max(enumerate(pred_args), key=lambda x: len(x))
                start = pred_in_trans_position[max_index]    
                filter_srl.append({'pred': pred_value, 'position': [start, start + len(pred_value) - 1], 'arguments': longest_args})           
                if pred_value in pred_visit:
                    pred_visit[pred_value].append(start)
                else:
                    pred_visit[pred_value] = [start]
        return {'text': trans, 'srl': filter_srl}
    else:
        return {'text': ' ', 'srl': []}