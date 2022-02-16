from scripts import settings
from scripts import utils
import time
import os

start_time = time.time()

all_params, all_scores = {}, {}
params_cols, scores_cols = [], []
overall_count = len(settings.participants)
counter = 1
for participant in settings.participants:
    print(f'({counter}/{overall_count}) {participant}', end=', test : ', flush=True)
    counter += 1

    all_params[participant] = []
    all_scores[participant] = []
    try:
        for params, scores in utils.participant_train_test_xgboost(participant=participant, train_dir=settings.ppg_filtered_dataset_dir, m=settings.m, model_dir=settings.ppg_filtered_model_dir):
            if len(params_cols) + len(scores_cols) == 0:
                params_cols = list(params.keys())
                params_cols.sort()
                scores_cols = list(scores.keys())
                scores_cols.sort()

            all_params[participant] += [params]
            all_scores[participant] += [scores]
    except ValueError:
        print('error')
        continue
    print()

with open(f"{settings.ppg_filter_test_results_dir}/scores.csv", "w+") as w_scores:
    with open(f"{settings.ppg_filter_test_results_dir}/params.csv", "w+") as w_params:
        w_scores.write('Participant,{}\n'.format(','.join(scores_cols)))
        w_params.write('Participant,{}\n'.format(','.join(params_cols)))
        for participant in all_params:
            for sub_params, sub_scores in zip(all_params[participant], all_scores[participant]):
                w_params.write(participant)
                w_scores.write(participant)
                for param_col in params_cols:
                    w_params.write(',{}'.format(sub_params[param_col]))
                for score_col in scores_cols:
                    w_scores.write(',{}'.format(sub_scores[score_col]))
                w_params.write('\n')
                w_scores.write('\n')
        w_params.close()
        w_scores.close()

print(' --- execution time : %s seconds --- ' % (time.time() - start_time))
