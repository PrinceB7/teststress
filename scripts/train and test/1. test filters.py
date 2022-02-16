from scripts import settings
from scripts import utils
import time
import os

focus_participant = 'azizsambo58@gmail.com'


def run_train_test(dataset_dir, result_dir, tune_parameters=False):
    all_params, all_scores = {}, {}
    params_cols, scores_cols = [], []
    all_files = [el for el in os.listdir(dataset_dir) if el.endswith('.csv')]
    if len(all_files) == 0 and os.path.exists(f'{dataset_dir}/1'):
        all_files = [el for el in os.listdir(f'{dataset_dir}/1') if el.endswith('.csv')]
    overall_count = len(all_files)
    counter = 1
    for filename in all_files:
        participant = filename[:-4]
        if participant != focus_participant:
            continue
        print(f'({counter}/{overall_count}) {participant}')
        counter += 1

        all_params[participant] = []
        all_scores[participant] = []
        try:
            for params, scores in utils.participant_train_test_xgboost(participant=participant, train_dir=dataset_dir, m=settings.m, tune_parameters=tune_parameters):
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

    with open(f"{result_dir}/scores.csv", "w+") as w_scores:
        with open(f"{result_dir}/params.csv", "w+") as w_params:
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


def main():
    start_time = time.time()

    run_train_test(dataset_dir=settings.not_filtered_dataset_dir, result_dir=settings.no_filter_test_results_dir)
    # run_train_test(dataset_dir=settings.ppg_filtered_dataset_dir, result_dir=settings.ppg_filter_test_results_dir, tune_parameters=False)
    # run_train_test(dataset_dir=settings.acc_filtered_dataset_dir, result_dir=settings.acc_filter_test_results_dir, tune_parameters=True)
    # run_train_test(dataset_dir=settings.combined_filtered_dataset_dir, result_dir=settings.combined_filter_test_results_dir, tune_parameters=True)

    print(' --- execution time : %s seconds --- ' % (time.time() - start_time))


if __name__ == '__main__':
    main()
