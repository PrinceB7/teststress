from scripts import settings
import os

combinations = [el[:-11] for el in os.listdir(settings.behavioral_data_combinations_test_results_dir) if el.endswith('_scores.csv')]
f1s = {}
roc_aucs = {}

for moments in combinations:
    f1s[moments] = []
    roc_aucs[moments] = []
    with open(f'{settings.behavioral_data_combinations_test_results_dir}/{moments}_scores.csv', 'r') as r:
        for line in r.readlines()[1:]:
            participant, acc, f1, roc_auc, tnr, tpr = line[:-1].split(',')
            f1s[moments] += [float(f1)]
            roc_aucs[moments] += [float(roc_auc)]
    f1s[moments] = sum(f1s[moments]) / len(f1s[moments])
    roc_aucs[moments] = sum(roc_aucs[moments]) / len(roc_aucs[moments])

print(f1s)
print(roc_aucs)
