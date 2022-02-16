from scripts import settings
from scripts import utils

thresholds = {
    'PPG': {
        'azizsambo58@gmail.com': 140000,
        'laurentkalpers3@gmail.com': 140000,
        'nazarov7mu@gmail.com': 80000,
        'jskim@nsl.inha.ac.kr': 99333,
        'aliceblackwood123@gmail.com': 110000,
        'mr.khikmatillo@gmail.com': 100000,
        'jumabek4044@gmail.com': 121000,
        'nnarziev@gmail.com': 130000,
        'nslabinha@gmail.com': 120000,
        'salman@nsl.inha.ac.kr': 125333,
    },
    'ACC': {
        'azizsambo58@gmail.com': 2.0,
        'laurentkalpers3@gmail.com': 2.0,
        'nazarov7mu@gmail.com': 1.47,
        'jskim@nsl.inha.ac.kr': 2.13,
        'aliceblackwood123@gmail.com': 1.9,
        'mr.khikmatillo@gmail.com': 2.2,
        'jumabek4044@gmail.com': 1.3,
        'nnarziev@gmail.com': 1.6,
        'nslabinha@gmail.com': 1.4,
        'salman@nsl.inha.ac.kr': 1.68,
    }
}

participant_counter = 1
for participant in thresholds['PPG']:
    sample_counter = 1
    # open(f'{settings.combined_filter_stats_dir}/{participant}.csv', 'w+') as w
    with open(f'{settings.filtered_ground_truths_file}', 'w+') as w_ema:
        # w.write('#,Timestamp,IBI,GT STRESS,LOOSENESS,REMOVED,MOBILITY,REMOVED\n')
        w_ema.write('"participant","is_self_report","timestamp","lose_control","difficult","confident","your_way","likert_stress_level"\n')

        print(f"{participant_counter}. {participant}; stress-feature calculation using IBI readings")
        participant_counter += 1

        participants_rr_dataset = utils.load_rr_data(participant=participant)
        participants_ground_truths = utils.load_ground_truths(participant=participant)
        participants_ppg_dataset = utils.load_ppg_data(participant=participant)
        participants_acc_dataset = utils.load_acc_data(participant=participant)

        for gt in participants_ground_truths:
            is_self_report = gt[0]
            timestamp = gt[1]

            gt_label = 1 if gt[-1] else 0
            # print(f'processing {participant}\'s GT at {timestamp}')
            till_timestamp = timestamp - (1800000 if is_self_report else settings.dataset_augmentation_window_sizes[participant])

            is_ppg_artifact = True
            is_acc_artifact = True
            while timestamp > till_timestamp and (is_ppg_artifact or is_acc_artifact):
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
                        ppg_stdev = utils.get_ppg_stdev(ppg_values=selected_ppg_values)
                        acc_magnitudes = utils.get_activeness_scores(selected_acc_values)

                        is_ppg_artifact = 1 if ppg_stdev >= thresholds['PPG'][participant] else 0
                        is_acc_artifact = 1 if utils.is_acc_window_active(activeness_scores=acc_magnitudes, activity_threshold=thresholds['ACC'][participant]) else 0

                        # w.write('{number},{timestamp},{avg_rr_interval},{gt_label},{ppg_stdev},{is_ppg_artifact},{avg_acc_magnitude},{is_acc_artifact}\n'.format(
                        #     timestamp=timestamp,
                        #     number=sample_counter,
                        #     avg_rr_interval=utils.average(values=selected_rr_intervals),
                        #     gt_label=gt_label,
                        #     ppg_stdev=ppg_stdev,
                        #     is_ppg_artifact=is_ppg_artifact,
                        #     avg_acc_magnitude=utils.average(values=acc_magnitudes),
                        #     is_acc_artifact=is_acc_artifact,
                        # ))
                        sample_counter += 1
                    except ValueError as e:
                        print('erroneous case met :', participant)
                        print(e)
                        pass

            if not is_ppg_artifact and not is_acc_artifact:
                lose_control = gt[2]
                difficult = gt[3]
                confident = gt[4]
                your_way = gt[5]
                likert_stress_level = gt[6]
                w_ema.write(f'{participant},{is_self_report},{timestamp},{lose_control},{difficult},{confident},{your_way},{likert_stress_level}\n')
