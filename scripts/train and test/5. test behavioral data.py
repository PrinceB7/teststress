from scripts import settings
from scripts import utils
import time
import os

start_time = time.time()

all_params, all_scores = {}, {}
params_cols, scores_cols = [], []
counter = 1
for filename in [el for el in os.listdir(settings.behavioral_features_dataset_dir) if el.endswith('.csv')]:
    participant = filename[:-4]
    print(f'{counter}. {participant}')
    counter += 1

    all_params[participant] = []
    all_scores[participant] = []
    params, scores = utils.participant_train_test_xgboost(participant=participant, train_dir=settings.behavioral_features_dataset_dir, selected_features=settings.behavior_enhanced_selected_features, tuning=True)

    if len(params_cols) + len(scores_cols) == 0:
        params_cols = list(params.keys())
        params_cols.sort()
        scores_cols = list(scores.keys())
        scores_cols.sort()

    all_params[participant] = params
    all_scores[participant] = scores

with open(f"{settings.behavioral_data_test_results_dir}/scores.csv", "w+") as w_scores, open(f"{settings.behavioral_data_test_results_dir}/params.csv", "w+") as w_params:
    w_params.write('Participant,{}\n'.format(','.join(params_cols)))
    w_scores.write('Participant,{}\n'.format(','.join(scores_cols)))
    for participant in all_params:
        w_params.write(participant)
        w_scores.write(participant)
        for param_col in params_cols:
            w_params.write(',{}'.format(all_params[participant][param_col]))
        for score_col in scores_cols:
            w_scores.write(',{}'.format(all_scores[participant][score_col]))
        w_params.write('\n')
        w_scores.write('\n')
print('completed!')
print(' --- execution time : %s seconds --- ' % (time.time() - start_time))
