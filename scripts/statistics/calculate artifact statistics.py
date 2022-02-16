from scripts import settings
from scripts import utils

counter = 1
with open(f'{settings.statistics_dir}/artifact_stats.csv', 'w+') as w:
    w.write('participant,ppg artifacts,ppg & acc artifacts,acc artifacts,clean,amount\n')
    for participant in settings.participants:
        print(f"{counter}. counting artifacts in {participant}'s dataset")
        counter += 1

        participants_rr_dataset = utils.load_rr_data(participant=participant)
        participants_ground_truths = utils.load_ground_truths(participant=participant)
        participants_ppg_dataset = utils.load_ppg_data(participant=participant)
        participants_acc_dataset = utils.load_acc_data(participant=participant)

        ppg_artifacts_count = 0
        acc_artifacts_count = 0
        ppg_acc_artifacts_count = 0
        clean_count = 0

        for ground_truth in participants_ground_truths:
            is_self_report = ground_truth[0]
            timestamp = ground_truth[1]
            # print(f'processing {participant}\'s GT at {timestamp}')
            till_timestamp = timestamp - (1800000 if is_self_report else settings.dataset_augmentation_window_size)

            while timestamp > till_timestamp:
                selected_rr_intervals = utils.select_data(
                    dataset=participants_rr_dataset,
                    from_ts=timestamp - settings.feature_aggregation_window_size,
                    till_ts=timestamp
                )
                selected_ppg_values = utils.select_data(
                    dataset=participants_ppg_dataset,
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

                if selected_rr_intervals is not None and selected_ppg_values is not None and selected_acc_values is not None:
                    try:
                        is_ppg_artifact = utils.get_ppg_stdev(ppg_values=selected_ppg_values) >= settings.combined_thresholds['PPG'][participant]
                        is_acc_artifact = utils.is_acc_window_active(acc_values=selected_acc_values, activity_threshold=settings.combined_thresholds['ACC'][participant])
                        if is_ppg_artifact and is_acc_artifact:  # ppg & acc artifact
                            ppg_acc_artifacts_count += 1
                        elif is_ppg_artifact:  # ppg artifact
                            ppg_artifacts_count += 1
                        elif is_acc_artifact:  # acc artifact
                            acc_artifacts_count += 1
                        else:  # clean
                            clean_count += 1
                    except ValueError as e:
                        print('erroneous case met :', participant)
                        print(e)
                        pass

        w.write(f'{participant},{ppg_artifacts_count},{ppg_acc_artifacts_count},{acc_artifacts_count},{clean_count}\n')
        print(ppg_artifacts_count + acc_artifacts_count + ppg_acc_artifacts_count + clean_count)
