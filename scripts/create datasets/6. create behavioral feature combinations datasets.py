from scripts import settings
from scripts import utils
import os
import time

start_time = time.time()

combinations = utils.calculate_combinations(array=[1, 2, 3, 4])
overall_count = len(settings.participants) * len(combinations)
counter = 1
participants_acc_datasets = {}
for moments in combinations:
    moments_directory = f"{settings.behavioral_features_combinations_dir}/{','.join([str(moment) for moment in moments])}"
    if not os.path.exists(moments_directory):
        os.mkdir(moments_directory)
    for participant in settings.participants:
        sub_start_time = time.time()
        with open(f'{moments_directory}/{participant}.csv', 'w+') as w, open(f'{settings.combined_filtered_dataset_dir}/{participant}.csv', 'r') as r:
            w.write(utils.get_behavioral_dataset_header(moments=moments))

            print(f"{counter}/{overall_count}. {participant}; behavioral-feature calculation using acceleration-signal readings")
            counter += 1

            if participant not in participants_acc_datasets:
                participants_acc_dataset = utils.load_acc_data(participant=participant)
                participants_acc_datasets[participant] = participants_acc_dataset
            else:
                participants_acc_dataset = participants_acc_datasets[participant]

            for line in r.readlines()[1:]:
                cells = line[:-1].split(',')
                stress_feature_values = ','.join(cells[:-9])
                ground_truth_values = ','.join(cells[-9:])
                timestamp = int(cells[0])
                # print(f'processing {participant}\'s features at {timestamp}')

                selected_acc_values = utils.select_data(
                    dataset=participants_acc_dataset,
                    from_ts=timestamp,
                    till_ts=timestamp + settings.feature_aggregation_window_size
                )
                if selected_acc_values is not None:
                    behavioral_features = utils.calculate_behavioral_features(selected_acc_values, moments=moments)
                    w.write(f"{stress_feature_values},{','.join([str(value) for value in behavioral_features])},{ground_truth_values}\n")
        print('--- %s seconds --- ' % (time.time() - sub_start_time))
print('completed!')
print(' --- overall execution time : %s seconds --- ' % (time.time() - start_time))
