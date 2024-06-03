import json
import copy
path = ''
path1 = ''
save_path = ''
with open(path, 'r', encoding='utf-8') as f, open(path1, 'r', encoding='utf-8') as f1, open(save_path, 'w', encoding='utf-8') as fout:
    for line, line1 in zip(f, f1):
        data = json.loads(line)
        fout.write(json.dumps(data, ensure_ascii=False)+'\n')
        data1 = json.loads(line1)
        new_data = copy.deepcopy(data)
        new_data['sentence'] = data1['text']
        new_data['srl'] = data1['srl']   
        fout.write(json.dumps(new_data, ensure_ascii=False)+'\n')     