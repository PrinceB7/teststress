from scripts import settings
from scripts import utils

counter = 1
for participant in settings.participants:
    with open(f'{settings.behavioral_features_dataset_dir}/{participant}.csv', 'w+') as w, open(f'{settings.combined_filtered_dataset_dir}/{participant}.csv', 'r') as r:
        w.write(settings.behavioral_dataset_header)

        print(f"{counter}. {participant}; behavioral-feature calculation using acceleration-signal readings")
        counter += 1

        participants_acc_dataset = utils.load_acc_data(participant=participant)

        for line in r.readlines()[1:]:
            cells = line[:-1].split(',')
            stress_feature_values = ','.join(cells[:-9])
            ground_truth_values = ','.join(cells[-9:])
            timestamp = int(cells[0])
            print(f'processing {participant}\'s features at {timestamp}')

            selected_acc_values = utils.select_data(
                dataset=participants_acc_dataset,
                from_ts=timestamp,
                till_ts=timestamp + settings.feature_aggregation_window_size
            )
            if selected_acc_values is not None:
                behavioral_features = utils.calculate_behavioral_features(selected_acc_values)
                w.write(f"{stress_feature_values},{','.join([str(value) for value in behavioral_features])},{ground_truth_values}\n")

# print stats for features - acc filter
print('participant-email\t\tsamples\t\tstressed\tnot-stressed')
for participant in settings.participants:
    with open('{0}/{1}.csv'.format(settings.behavioral_features_dataset_dir, participant), 'r') as r:
        lines = r.readlines()[1:]
        print('{0}\t\t{1}\t\t{2}\t\t{3}'.format(
            participant[:20],
            len(lines),
            len([1 for line in lines if line[:-1].split(',')[-1].lower() == 'true']),
            len([1 for line in lines if not line[:-1].split(',')[-1].lower() == 'true'])
        ))
