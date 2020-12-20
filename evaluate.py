import os
import json
import numpy as np
from metric import metrics

prefix = 'test'
count_mark = '15'
epoch = 2
dataset = 'test'
ans_max = 2
suffix = ''
loss = 'dsloss'
do_multi = 0
do_calibration = True

test_data_path = 'dataset/NYT/%s/%s_%s%s.json' % (count_mark,count_mark, dataset, suffix)

with open(test_data_path, 'r') as f:
    dic = json.load(f)

obj = {}
tokens1 = []
tokens2 = []
for key, value in dic.items():
    document = value['document']
    head = value['head']
    entities = [v['text'] for v in value['entities']]
    assert entities[0] == 'NA'
    entities = entities[1:]
    for q in value['qas']:
        ans = set()
        pos = set()
        for a in q['answers']:
            ans.add(a['text'])
            pos.update(a['answer_starts'])
        ans = list(ans)
        obj[q['id']] = (q['question'], head, entities, ans, document)

if dataset == 'dev':
    data_path = 'checkpoints/nyt_bert_%s%s_%s/nbest_predictions_%d.json' % (loss, suffix, count_mark, epoch)
else:
    if epoch < 0:
        data_path = 'checkpoints/nyt_bert_%s%s_%s/nbest_predictions_test.json' % (loss, suffix, count_mark)
    else:
        data_path = 'checkpoints/nyt_bert_%s%s_%s/checkpoint-epoch-%d/nbest_predictions_test.json' % (loss, suffix, count_mark, epoch)

with open(data_path, 'r') as f:
    data = json.load(f)

multi_pairs = set()
with open('dataset/multi_pairs.txt', 'r') as f:
    for line in f:
        h, t = line.strip().split('\t')
        multi_pairs.add((h, t))

predictions = {}
for key, values in data.items():
    text = values[0]['text']
    prob = values[0]['probability']
    tps = []
    for i in range(len(values)):
        tps.append((values[i]['text'], values[i]['probability']))
    tpss = sorted(tps, key=lambda x: x[1], reverse=True)
    text_map = {}
    texts = []
    probs = []
    for t, p in tpss:
        if t not in text_map:
            texts.append(t)
            probs.append(p)
            text_map[t] = len(text_map)
        '''
        else:
            index = text_map[t]
            probs[index] = probs[index] + p
        '''
    all_prob = sum(probs)
    for i in range(len(probs)):
        probs[i] = probs[i] / all_prob
    tps = []
    for t, p in zip(texts, probs):
        tps.append((t, p))
    tpss = sorted(tps, key=lambda x: x[1], reverse=True)
    texts = [t[0] for t in tpss]
    probs = [t[1] for t in tpss]
    predictions[key] = (texts, probs)

sorted_preds = sorted(predictions.items(), key=lambda x: x[1][1], reverse=True)

pn_s = []
pn_l = []
n_match, n_predict, n_true = 0, 0, 0
n_na_positive = 0
n_na_positive_potential = 0
n_false_positive = 0
n_neg_positive = 0
n_false_na = 0

false_pos = []
count_pos = []
false_neg = []
count_neg = []
false_pe = []
count_pe = []
true_pos = []
count_true = []
true_pos = []
_id = -1
y_trues = []
y_scores = []

for key, (texts, probs) in sorted_preds:
    text = texts[0]
    prob = probs[0]
    question = obj[key][0]
    head = obj[key][1]
    targets = obj[key][3]
    document = obj[key][4]
    y_true = [0] * ans_max
    y_score = [-1.] * ans_max
    idx = 1
    if len(targets) <= 0:
        y_true[0] = 1
    count = 0
    na_pp = -1
    other_pp = 0
    for tt, pp in zip(texts, probs):
        if not tt:
            na_pp = pp
            other_pp += na_pp
        elif na_pp >= 0 and pp <= na_pp:
            other_pp += pp
    for tt, pp in zip(texts, probs):
        if not tt:
            y_score[0] = pp
        else:
            if do_multi > 0:
                if do_multi == 1 and (head, tt) in multi_pairs:
                    continue
                if do_multi == 2 and (head, tt) not in multi_pairs:
                    continue
            if do_calibration and pp > na_pp:
                pp = pp / (pp + na_pp)
            if tt in targets:
                y_true[idx] = 1
            y_score[idx] = pp
            idx += 1
    y_trues.append(y_true)
    y_scores.append(y_score)
    if len(targets) > 0:
        n_true += 1
    if text:
        _id += 1
        n_predict += 1
        if text in targets:
            n_match += 1
            count_true.append(len(document.split(' ')))
            tokens1.append(len(list(set(texts))))
        else:
            n_neg_positive += 1
            tokens2.append(len(list(set(texts))))
            if targets:
                n_false_positive += 1
                count_pe.append(len(document.split(' ')))
            else:
                n_na_positive += 1
                count_pos.append(len(document.split(' ')))
                if texts[1] == '':
                    n_na_positive_potential += 1

        if n_predict in [100, 200, 300, 400, 500]:
            pn_s.append(n_match / n_predict)
            pn_l.append('P@%d' % n_predict)
    else:
        if text in targets:
            print(targets)
        if len(targets) > 0:
            n_false_na += 1
            count_neg.append(len(document.split(' ')))

print('======precision-recall-curve======')
y_scores = np.array(y_scores)
y_trues = np.array(y_trues)

if dataset == 'test':
    print((y_scores[:, 1:] >= 0).sum())
    print(y_trues[:, 1:].sum())
metrics(y_scores[:, 1:], y_trues[:, 1:], prefix, do_reshape=True)
