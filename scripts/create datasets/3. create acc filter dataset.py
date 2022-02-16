from scripts import settings
from scripts import utils

counter = 1
for participant in settings.participants:
    with open(f'{settings.acc_filtered_dataset_dir}/{participant}.csv', 'w+') as w:
        w.write(settings.filter_datasets_header)

        print(f"{counter}. {participant}; stress-feature calculation using IBI readings")
        counter += 1

        participants_rr_dataset = utils.load_rr_data(participant=participant)
        participants_ground_truths = utils.load_ground_truths(participant=participant)
        participants_acc_dataset = utils.load_acc_data(participant=participant)

        for ground_truth in participants_ground_truths:
            is_self_report = ground_truth[0]
            timestamp = ground_truth[1]
            print(f'processing {participant}\'s GT at {timestamp}')
            till_timestamp = timestamp - (1800000 if is_self_report else settings.dataset_augmentation_window_size)

            while timestamp > till_timestamp:
                selected_rr_intervals = utils.select_data(
                    dataset=participants_rr_dataset,
                    from_ts=timestamp - settings.feature_aggregation_window_size,
                    till_ts=timestamp
                )
                selected_acc_values = utils.select_data(
                    dataset=participants_acc_dataset,
                    from_ts=timestamp - settings.feature_aggregation_window_size,
                    till_ts=timestamp,
                    with_timestamp=True
                )
                timestamp -= settings.feature_aggregation_window_size

                if selected_rr_intervals is not None and selected_acc_values is not None:
                    try:
                        if not utils.is_acc_window_active(acc_values=selected_acc_values, activity_threshold=settings.acc_thresholds[participant]):
                            features = utils.calculate_features(selected_rr_intervals)
                            w.write('{0},{1},{2}\n'.format(
                                timestamp,
                                ','.join([str(value) for value in features]),
                                ','.join([str(value) for value in ground_truth])
                            ))
                    except ValueError as e:
                        print('erroneous case met :', participant)
                        print(e)
                        pass

# print stats for features - acc filter
print('participant-email\t\tsamples\t\tstressed\tnot-stressed')
for participant in settings.participants:
    with open('{0}/{1}.csv'.format(settings.acc_filtered_dataset_dir, participant), 'r') as r:
        lines = r.readlines()[1:]
        print('{0}\t\t{1}\t\t{2}\t\t{3}'.format(
            participant[:20],
            len(lines),
            len([1 for line in lines if line[:-1].split(',')[-1].lower() == 'true']),
            len([1 for line in lines if not line[:-1].split(',')[-1].lower() == 'true'])
        ))
