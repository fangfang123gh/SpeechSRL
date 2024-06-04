import json
import copy
path = ''
path1 = ''
save_path = ''
datas = {}
with open(path, 'r', encoding='utf-8') as f, open(path1, 'r', encoding='utf-8') as f1, open(save_path, 'w', encoding='utf-8') as fout:
    for line in f:
        data = json.loads(line)
        fout.write(json.dumps(data, ensure_ascii=False)+'\n')
        datas[data['sentence']] = data
    for line1 in f1:
        data1 = json.loads(line1)
        new_data = copy.deepcopy(datas[data1['gold_text']])
        new_data['sentence'] = data1['text']
        new_data['srl'] = data1['srl'] 
        

        fout.write(json.dumps(new_data, ensure_ascii=False)+'\n')    
