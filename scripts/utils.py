from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features, get_frequency_domain_features
from hrvanalysis import get_csi_cvi_features, get_geometrical_features
from hrvanalysis import get_poincare_plot_features, get_sampen
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from scipy.stats import moment
import plotly.graph_objs as go
from scripts import settings
import xgboost as xgb
import pandas as pd
import numpy as np
import statistics
import datetime
import math
import time
import csv
import os


# get timestamp in milliseconds
def get_timestamp_ms():
    return int(time.time() * 1000)


# string format validator : are_numeric
def are_numeric(*args):
    return False not in [is_numeric(value) for value in args]


# string format validator : is_numeric
def is_numeric(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


# string format validator : is_float
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


# converts string time into timestamp (i.e., 2020-04-23T11:00+0900 --> 1587607200000)
def string_to_timestamp(_str):
    if _str == '':
        return None
    elif _str[-3] == ':':
        _str = _str[:-3] + _str[-2:]
    return int(datetime.datetime.strptime(_str, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()) * 1000


# converts a numeric string into a number
def string_to_number(_str):
    if _str == '':
        return None
    else:
        try:
            return int(_str)
        except ValueError:
            return None


# converts a numeric string into a fraction number
def string_to_float(_str):
    if _str == '':
        return None
    else:
        try:
            return float(_str)
        except ValueError:
            return None


# loads ESM responses, calculates scores, and adds a label (i.e., "stressed" / "not-stressed")
def load_ground_truths(participant):
    res = []
    with open(settings.untouched_ground_truths_file, 'r') as r:
        csv_reader = csv.reader(r, delimiter=',', quotechar='"')
        for csv_row in csv_reader:
            if csv_row[0] != participant:
                continue
            rt = string_to_timestamp(csv_row[11])
            st = string_to_timestamp(csv_row[12])
            control = string_to_number(csv_row[16])
            difficulty = string_to_number(csv_row[17])
            confident = string_to_number(csv_row[18])
            yourway = string_to_number(csv_row[19])
            row = (
                st is None,  # is self report
                rt if st is None else st,  # timestamp
                control,  # (-)PSS:Control
                difficulty,  # (-)PSS:Difficult
                confident,  # (+)PSS:Confident
                yourway,  # (+)PSS:YourWay
                string_to_number(csv_row[20]),  # LikeRt:StressLevel,
            )
            if None in row:
                continue
            score = (control + difficulty + (6 - confident) + (6 - yourway)) / 4
            res += [row + (score,)]
        res.sort(key=lambda e: e[1])
        mean = sum(row[7] for row in res) / len(res)
        for i in range(len(res)):
            res[i] += (res[i][7] > mean,)
    return res


# load ESM responses after combined filter was applied
def load_unfiltered_ground_truths(participant):
    res = []
    with open(settings.untouched_ground_truths_file, 'r') as r:
        pass
    return res


# loads participant's IBI readings
def load_rr_data(participant):
    res = []
    file_path = os.path.join(settings.raw_dataset_dir, participant, 'RR_INTERVAL.csv')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as r:
        for line in r:
            ts, rr = line[:-1].split(',')
            try:
                ts, rr = int(ts), int(rr)
            except ValueError:
                continue
            res += [(ts, rr)]
        res.sort(key=lambda e: e[0])
    return res


# loads participant's ACC readings
def load_acc_data(participant):
    res = []
    file_path = os.path.join(settings.raw_dataset_dir, participant, 'ACCELEROMETER.csv')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as r:
        for line in r:
            cells = line[:-1].split(',')
            if len(cells) == 2:
                continue
            try:
                ts, x, y, z = int(cells[0]), float(cells[1]), float(cells[2]), float(cells[3])
            except ValueError:
                continue
            res += [(ts, x, y, z)]
        res.sort(key=lambda e: e[0])
    return res


# loads participant's PPG light intensity readings
def load_ppg_data(participant):
    res = []
    file_path = os.path.join(settings.raw_dataset_dir, participant, 'LIGHT_INTENSITY.csv')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as r:
        for line in r:
            ts, rr = line[:-1].split(',')
            try:
                ts, rr = int(ts), int(rr)
            except ValueError:
                continue
            res += [(ts, rr)]
        res.sort(key=lambda e: e[0])
    return res


# calculate STD of ppg light intensities
def get_ppg_stdev(ppg_values):
    return statistics.stdev(ppg_values)


# searches for element in dataset with binary search
def bin_find(dataset, low, high, ts):
    if high > low:
        mid = (high + low) // 2
        if dataset[mid][0] == ts:
            return mid
        elif dataset[mid][0] > ts:
            return bin_find(dataset, low, mid, ts)
        else:
            return bin_find(dataset, mid + 1, high, ts)
    else:
        return min(high, low)


# selects data within the specified timespan
def select_data(dataset, from_ts, till_ts, with_timestamp=False):
    res = []
    index = bin_find(dataset, 0, len(dataset) - 1, from_ts)
    end_index = len(dataset)
    while index < end_index:
        if from_ts <= dataset[index][0] < till_ts:
            if with_timestamp:
                res += [tuple(dataset[index])]
            elif len(dataset[index][1:]) == 1:
                res += tuple(dataset[index][1:])
            else:
                res += [tuple(dataset[index][1:])]
        elif dataset[index][0] >= till_ts:
            break
        index += 1
    return res if len(res) >= 60 else None


# selects the closest data point (w/ timestamp)
def find_closest(start_ts, ts, dataset):
    while ts not in dataset and ts != start_ts:
        ts -= 1
    if ts == start_ts:
        # print('error occurred, reached the start_ts!')
        # exit(1)
        return None
    else:
        return dataset[ts]


# downsamples data
def select_downsample_data(dataset, from_ts, till_ts):
    selected_data = {}
    for row in dataset:
        if from_ts <= row[0] < till_ts:
            selected_data[row[0]] = row[1]
        elif row[0] >= till_ts:
            break
    timestamps = [ts for ts in range(from_ts, till_ts, 1000)]
    res = []
    for ts in timestamps:
        closest = find_closest(start_ts=from_ts, ts=ts, dataset=selected_data)
        if closest is not None:
            res += [closest]
    return res if len(res) >= 20 else None


# calculates activeness score of acc window of data
def get_activeness_scores(acc_values):
    acc_magnitudes = []
    magnitudes = []
    end = acc_values[0][0] + settings.activity_window_size
    last_ts = acc_values[-1][0]
    for ts, x, y, z in acc_values:
        if ts >= end or ts == last_ts:
            acc_magnitudes += [np.mean(magnitudes)]
            end += settings.activity_window_size
            magnitudes = []
        magnitudes += [math.sqrt(x ** 2 + y ** 2 + z ** 2)]
    # low_limit = np.percentile(stdevs, 1)
    # high_limit = np.percentile(stdevs, 99)
    # activeness_range = high_limit - low_limit
    return acc_magnitudes


# checks if window of accelerations is considered as active
def is_acc_window_active(activity_threshold, acc_values=None, activeness_scores=None):
    if acc_values is not None:
        activeness_scores = get_activeness_scores(acc_values=acc_values)
    active_count = 0
    for activeness_score in activeness_scores:
        # if activeness_score > (low_limit + activity_threshold * activeness_range):
        if activeness_score > activity_threshold:
            active_count += 1
    if active_count > len(activeness_scores) / 2:
        return True
    else:
        return False


# calculates stress features from provided IBI readings
def calculate_features(rr_intervals):
    # process the RR-intervals
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals, low_rri=300, high_rri=2000, verbose=False)
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method='linear')
    nn_intervals = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method='malik')
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals)

    # extract the features
    time_domain_features = get_time_domain_features(nn_intervals=interpolated_nn_intervals)
    frequency_domain_features = get_frequency_domain_features(nn_intervals=interpolated_nn_intervals)
    csi_cvi_features = get_csi_cvi_features(nn_intervals=interpolated_nn_intervals)
    geometrical_features = get_geometrical_features(nn_intervals=interpolated_nn_intervals)
    poincare_plot_features = get_poincare_plot_features(nn_intervals=interpolated_nn_intervals)
    sample_entropy = get_sampen(nn_intervals=interpolated_nn_intervals)

    return [
        time_domain_features['mean_nni'],  # The mean of RR-intervals
        time_domain_features['sdnn'],  # The standard deviation of the time interval between successive normal heart beats (i.e. the RR-intervals)
        time_domain_features['sdsd'],  # The standard deviation of differences between adjacent RR-intervals
        time_domain_features['rmssd'],  # The square root of the mean of the sum of the squares of differences between adjacent NN-intervals. Reflects high frequency (fast or parasympathetic) influences on hrV (i.e., those influencing larger changes from one beat to the next)
        time_domain_features['median_nni'],  # Median Absolute values of the successive differences between the RR-intervals
        time_domain_features['nni_50'],  # Number of interval differences of successive RR-intervals greater than 50 ms
        time_domain_features['pnni_50'],  # The proportion derived by dividing nni_50 (The number of interval differences of successive RR-intervals greater than 50 ms) by the total number of RR-intervals
        time_domain_features['nni_20'],  # Number of interval differences of successive RR-intervals greater than 20 ms
        time_domain_features['pnni_20'],  # The proportion derived by dividing nni_20 (The number of interval differences of successive RR-intervals greater than 20 ms) by the total number of RR-intervals
        time_domain_features['range_nni'],  # Difference between the maximum and minimum nn_interval
        time_domain_features['cvsd'],  # Coefficient of variation of successive differences equal to the rmssd divided by mean_nni
        time_domain_features['cvnni'],  # Coefficient of variation equal to the ratio of sdnn divided by mean_nni
        time_domain_features['mean_hr'],  # Mean heart rate value
        time_domain_features['max_hr'],  # Maximum heart rate value
        time_domain_features['min_hr'],  # Minimum heart rate value
        time_domain_features['std_hr'],  # Standard deviation of heart rate values

        frequency_domain_features['total_power'],  # Total power density spectral
        frequency_domain_features['vlf'],  # variance (=power) in HRV in the Very low Frequency (.003 to .04 Hz by default). Reflect an intrinsic rhythm produced by the heart which is modulated primarily by sympathetic activity
        frequency_domain_features['lf'],  # variance (=power) in HRV in the Low Frequency (.04 to .15 Hz). Reflects a mixture of sympathetic and parasympathetic activity, but in long-term recordings, it reflects sympathetic activity and can be reduced by the beta-adrenergic antagonist propanolol
        frequency_domain_features['hf'],  # variance (=power) in HRV in the High Frequency (.15 to .40 Hz by default). Reflects fast changes in beat-to-beat variability due to parasympathetic (vagal) activity. Sometimes called the respiratory band because it corresponds to HRV changes related to the respiratory cycle and can be increased by slow, deep breathing (about 6 or 7 breaths per minute) and decreased by anticholinergic drugs or vagal blockade
        frequency_domain_features['lf_hf_ratio'],  # lf/hf ratio is sometimes used by some investigators as a quantitative mirror of the sympatho/vagal balance
        frequency_domain_features['lfnu'],  # normalized lf power
        frequency_domain_features['hfnu'],  # normalized hf power

        csi_cvi_features['csi'],  # Cardiac Sympathetic Index
        csi_cvi_features['cvi'],  # Cardiac Vagal Index
        csi_cvi_features['Modified_csi'],  # Modified CSI is an alternative measure in research of seizure detection

        geometrical_features['triangular_index'],  # The HRV triangular index measurement is the integral of the density distribution (= the number of all NN-intervals) divided by the maximum of the density distribution
        geometrical_features['tinn'],  # The triangular interpolation of NN-interval histogram (TINN) is the baseline width of the distribution measured as a base of a triangle, approximating the NN-interval distribution

        poincare_plot_features['sd1'],  # The standard deviation of projection of the Poincaré plot on the line perpendicular to the line of identity
        poincare_plot_features['sd2'],  # SD2 is defined as the standard deviation of the projection of the Poincaré plot on the line of identity (y=x)
        poincare_plot_features['ratio_sd2_sd1'],  # Ratio between SD2 and SD1

        sample_entropy['sampen'],  # The sample entropy of the Normal to Normal Intervals
    ]


# calculate behavioral features from provided acceleration readings
def calculate_behavioral_features(acc_values, moments=None):
    acc_magnitudes = [math.sqrt(x ** 2 + y ** 2 + z ** 2) for x, y, z in acc_values]
    if moments is None:
        return [moment(acc_magnitudes, moment=_moment) for _moment in range(1, 5)]
    else:
        return [moment(acc_magnitudes, moment=_moment) for _moment in moments]


# makes suitable behavioral dataset header for the given moments
def get_behavioral_dataset_header(moments):
    moments_map = {1: '1st_moment', 2: '2nd_moment', 3: '3rd_moment', 4: '4th_moment'}
    return f"timestamp,{','.join(settings.all_features)},{','.join([moments_map[moment] for moment in moments])},gt_self_report,gt_timestamp,gt_pss_control,gt_pss_difficult,gt_pss_confident,gt_pss_yourway,gt_likert_stresslevel,gt_score,gt_label\n"


# makes suitable behavioral columns to select
def get_behavior_enhanced_selected_features(moments_string):
    moments_map = {1: '1st_moment', 2: '2nd_moment', 3: '3rd_moment', 4: '4th_moment'}
    moments = [int(moment) for moment in moments_string.split(',')]
    return settings.basic_selected_features + [moments_map[moment] for moment in moments]


# calculates combinations of the elements in array
def calculate_combinations(array):
    def recursive_combinations(array):
        if len(array) == 1:
            return [[], [array[0]]]
        else:
            res = []
            for i, val in enumerate(array):
                sub_array = array[:i] + array[i + 1:]
                for combination in recursive_combinations(array=sub_array):
                    combination.sort()
                    if combination not in res:
                        res += [combination]
                    combination = [val] + combination
                    combination.sort()
                    if combination not in res:
                        res += [combination]
            return res

    combinations = recursive_combinations(array=array)
    combinations.remove([])
    return combinations


# loads dataset, excluding some samples if needed
def load_dataset(directory, filename, selected_column_names, screen_out_timestamps=None):
    if not os.path.exists(f'{directory}/{filename}'):
        return None

    dataset = pd.read_csv(f'{directory}/{filename}').replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    # .drop_duplicates(subset='timestamp')
    if screen_out_timestamps is not None:
        dataset = dataset[~dataset.timestamp.isin(screen_out_timestamps)]
    features = dataset[selected_column_names]
    label = dataset.gt_label.astype(int)

    return list(dataset.timestamp), features, label


# loads test datasets
def load_all_test_datasets(directory, selected_column_names=settings.baseline_test_dir, screen_out_timestamps=None):
    datasets = []
    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue
        datasets += [(load_dataset(directory=directory, filename=filename, selected_column_names=selected_column_names, screen_out_timestamps=screen_out_timestamps))]
    return datasets


# calculates the value for M (number of test datasets)
def calculate_number_of_test_datasets(participant):
    def are_valid(*args):
        return False not in [val is not None and val != '' for val in args]

    def are_not_none(*args):
        return None not in args

    unfiltered_ground_truths_count = 0
    unfiltered_samples_count = 0
    with open(settings.untouched_ground_truths_file, 'r') as r:
        rr_dataset = load_rr_data(participant=participant)
        for csv_row in csv.reader(r, delimiter=',', quotechar='"'):
            if csv_row[0] != participant:
                continue
            elif are_not_none(csv_row[11], csv_row[12], csv_row[16], csv_row[17], csv_row[18], csv_row[19]) and are_valid(csv_row[11], csv_row[12]) and are_numeric(csv_row[16], csv_row[17], csv_row[18], csv_row[19]):
                rt = string_to_timestamp(csv_row[11])
                st = string_to_timestamp(csv_row[12])
                is_self_report = st is None
                timestamp = rt if st is None else st
                till_timestamp = timestamp - (settings.dataset_augmentation_window_size if is_self_report else settings.dataset_augmentation_window_sizes[participant])
                while timestamp > till_timestamp:
                    selected_rr_intervals = select_data(
                        dataset=rr_dataset,
                        from_ts=timestamp - settings.feature_aggregation_window_size,
                        till_ts=timestamp
                    )
                    if selected_rr_intervals is not None:
                        unfiltered_samples_count += 1
                    timestamp -= settings.feature_aggregation_window_size
                unfiltered_ground_truths_count += 1

    with open(f'{settings.combined_filtered_dataset_dir}/{settings.combined_thresholds["PPG"][participant]}/{settings.combined_thresholds["ACC"][participant]}/{participant}.csv', 'r') as r:
        filtered_samples_count = len(r.readlines()) - 1
    hyp_filtered_ground_truths_count = int(math.ceil(unfiltered_ground_truths_count * float(filtered_samples_count / unfiltered_samples_count)))
    res = math.ceil(hyp_filtered_ground_truths_count / 4)
    # print(f'M = {hyp_filtered_ground_truths_count}/4 = {res}')
    return res


# trains and tests on specified test datasets
def participant_train_test_xgboost(participant, train_dir, m=None, test_dir=settings.baseline_test_dir, selected_feature_names=settings.basic_selected_features, tune_parameters=False, verbose=False, average=False, model_dir=None):
    test_datasets = []
    test_results = []
    if verbose:
        print('training and testing...')
    if m is None:
        m = calculate_number_of_test_datasets(participant=participant)
    for i in range(1, m + 1):
        test_dataset = load_dataset(directory=f'{test_dir}/{participant}', filename=f'{i}.csv', selected_column_names=selected_feature_names, screen_out_timestamps=None)
        if test_dataset is not None:
            test_datasets += [test_dataset]

    test_number = 1
    for ts, test_features, test_labels in test_datasets:
        print(test_number, end=' ', flush=True)
        test = True
        if os.path.exists(f'{train_dir}/{participant}.csv'):
            _, train_features, train_labels = load_dataset(
                directory=train_dir,
                filename=f'{participant}.csv',
                selected_column_names=selected_feature_names,
                screen_out_timestamps=ts
            )
        else:
            _, train_features, train_labels = load_dataset(
                directory=f'{train_dir}/{test_number}',
                filename=f'{participant}.csv',
                selected_column_names=selected_feature_names,
                screen_out_timestamps=ts
            )

        # configure test dataset
        scaler = MinMaxScaler()
        scaler.fit(test_features)
        test_features_scaled = scaler.transform(test_features)
        test_features = pd.DataFrame(test_features_scaled, index=test_features.index, columns=test_features.columns)

        k_folds = []
        splitter = StratifiedKFold(n_splits=5, shuffle=True)
        for idx, (train_indices, test_indices) in enumerate(splitter.split(train_features, train_labels)):
            try:
                x_train = train_features.iloc[train_indices]
                y_train = train_labels.iloc[train_indices]
                x_test = train_features.iloc[test_indices]
                y_test = train_labels.iloc[test_indices]
                k_folds.append((x_train, y_train, x_test, y_test))
            except ValueError:
                test = False
                continue

        # print('# Features : rows({rows}) cols({cols})'.format(rows=train_features.shape[0], cols=train_features.shape[1]))
        # print(train_features.head(), '\n')
        # print('# Labels : stressed({stressed}) not-stressed({not_stressed})'.format(stressed=np.count_nonzero(train_labels == 1), not_stressed=np.count_nonzero(train_labels == 0)))
        # print(train_labels.head(), '\n')

        k_folds_sampled = []
        for idx, (x_train, y_train, x_test, y_test) in enumerate(k_folds):
            try:
                sampler = SMOTE()
                x_sample, y_sample = sampler.fit_resample(x_train, y_train)
                x_sample = pd.DataFrame(x_sample, columns=x_train.columns)
                y_sample = pd.Series(y_sample)
                k_folds_sampled.append((x_sample, y_sample, x_test, y_test))
            except ValueError:
                test = False
                continue

        k_folds_scaled = []
        for x_train, y_train, x_test, y_test in k_folds_sampled:
            scaler = MinMaxScaler()
            scaler.fit(x_train)
            x_train_scale = scaler.transform(x_train)
            x_test_scale = scaler.transform(x_test)
            x_train = pd.DataFrame(x_train_scale, index=x_train.index, columns=x_train.columns)
            x_test = pd.DataFrame(x_test_scale, index=x_test.index, columns=x_test.columns)
            k_folds_scaled.append((x_train, y_train, x_test, y_test))

        conf_mtx = np.zeros((2, 2))  # 2 X 2 confusion matrix
        train_data = xgb.DMatrix(data=train_features, label=train_labels.to_numpy())

        # Parameter tuning / grid search
        train_parameters = {
            'max_depth': 6,
            'min_child_weight': 1,
            'eta': .3,
            'subsample': 1,
            'colsample_bytree': 1,
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'verbosity': 0,
            'eval_metric': "auc"
        }
        if tune_parameters:
            if verbose:
                print('tuning parameters...')

            temporary_parameters = {
                'max_depth': 6,
                'min_child_weight': 1,
                'eta': .3,
                'subsample': 1,
                'colsample_bytree': 1,
                'objective': 'binary:logistic',
                'booster': 'gbtree',
                'verbosity': 0,
                'eval_metric': "auc"
            }
            grid_search_params = [(max_depth, min_child_weight) for max_depth in range(0, 12) for min_child_weight in range(0, 8)]
            current_test_auc = -float("Inf")
            for max_depth, min_child_weight in grid_search_params:
                try:
                    # Update our parameters
                    temporary_parameters['max_depth'] = max_depth
                    temporary_parameters['min_child_weight'] = min_child_weight
                    # Run CV
                    cv_results = xgb.cv(temporary_parameters, train_data, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                    # Update best MAE
                    mean_mae = cv_results['test-auc-mean'].max()
                    if mean_mae > current_test_auc:
                        current_test_auc = mean_mae
                        train_parameters['max_depth'] = max_depth
                        train_parameters['min_child_weight'] = min_child_weight
                except xgb.core.XGBoostError:
                    continue

            grid_search_params = [(subsample, colsample) for subsample in [i / 10. for i in range(7, 11)] for colsample in [i / 10. for i in range(7, 11)]]
            current_test_auc = -float("Inf")
            temporary_parameters = {'subsample': None, 'colsample_bytree': None}
            # We start by the largest values and go down to the smallest
            for sub_sample, col_sample in reversed(grid_search_params):
                try:
                    # We update our parameters
                    train_parameters['subsample'] = sub_sample
                    train_parameters['colsample_bytree'] = col_sample
                    # Run CV
                    cv_results = xgb.cv(train_parameters, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                    mean_mae = cv_results['test-auc-mean'].max()
                    if mean_mae > current_test_auc:
                        current_test_auc = mean_mae
                        temporary_parameters = {'subsample': sub_sample, 'colsample_bytree': col_sample}
                except xgb.core.XGBoostError:
                    continue
            train_parameters['subsample'] = temporary_parameters['subsample']
            train_parameters['colsample_bytree'] = temporary_parameters['colsample_bytree']

            current_test_auc = -float("Inf")
            temporary_parameters = None
            for eta in [.3, .2, .1, .05, .01, .005]:
                try:
                    # We update our parameters
                    train_parameters['eta'] = eta
                    # Run and time CV
                    cv_results = xgb.cv(train_parameters, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                    # Update best score
                    mean_mae = cv_results['test-auc-mean'].max()
                    if mean_mae > current_test_auc:
                        current_test_auc = mean_mae
                        temporary_parameters = eta
                except xgb.core.XGBoostError:
                    continue
            train_parameters['eta'] = temporary_parameters

            current_test_auc = -float("Inf")
            temporary_parameters = None
            gamma_range = [i / 10.0 for i in range(0, 25)]
            for gamma in gamma_range:
                try:
                    # We update our parameters
                    train_parameters['gamma'] = gamma
                    # Run and time CV
                    cv_results = xgb.cv(train_parameters, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                    # Update best score
                    mean_mae = cv_results['test-auc-mean'].max()
                    if mean_mae > current_test_auc:
                        current_test_auc = mean_mae
                        temporary_parameters = gamma
                except xgb.core.XGBoostError:
                    continue
            train_parameters['gamma'] = temporary_parameters

        xgb_models = []  # This is used to store models for each fold.
        folds_scores_tmp = {'Accuracy (balanced)': [], 'F1 score': [], 'ROC AUC score': [], 'True Positive rate': [], 'True Negative rate': []}
        for x_train, y_train, x_test, y_test in k_folds_scaled:
            train_data = xgb.DMatrix(data=x_train, label=y_train.to_numpy())
            evaluation_data = xgb.DMatrix(data=x_test, label=y_test.to_numpy())
            test_data = xgb.DMatrix(data=test_features, label=test_labels.to_numpy())

            # docs : https://xgboost.readthedocs.io/en/latest/parameter.html
            results = {}
            try:
                booster = xgb.train(train_parameters, dtrain=train_data, num_boost_round=1000, early_stopping_rounds=25, evals=[(evaluation_data, 'test')], verbose_eval=False, evals_result=results)
                if test_number == settings.no_filter_best_tests[participant]:
                    model = booster
                    X = x_train
                    Y = y_train
            except xgb.core.XGBoostError:
                print(f'(test dataset #{test_number} error)', end=' ', flush=True)
                continue
            if verbose:
                print('Fold evaluation results : ', results)

            if test:
                predicted_probabilities = booster.predict(data=test_data, ntree_limit=booster.best_ntree_limit)
                predicted_labels = np.where(predicted_probabilities > 0.5, 1, 0)

                acc = balanced_accuracy_score(test_labels, predicted_labels)
                f1 = f1_score(test_labels, predicted_labels, average='macro')
                roc_auc = roc_auc_score(test_labels, predicted_probabilities)
                tpr = recall_score(test_labels, predicted_labels)
                tnr = recall_score(test_labels, predicted_labels, pos_label=0)

                folds_scores_tmp['Accuracy (balanced)'].append(acc)
                folds_scores_tmp['F1 score'].append(f1)
                folds_scores_tmp['ROC AUC score'].append(roc_auc)
                folds_scores_tmp['True Positive rate'].append(tpr)
                folds_scores_tmp['True Negative rate'].append(tnr)

                conf_mtx += confusion_matrix(test_labels, predicted_labels)
                xgb_models.append(booster)

        if test:
            folds_scores = {}
            for k, v in folds_scores_tmp.items():
                folds_scores[k] = np.mean(v)

            test_results += [[train_parameters, folds_scores]]
        test_number += 1

    if model_dir is not None:
        txt_model_path = f'{model_dir}/{participant}.txt'
        bin_model_path = f'{model_dir}/{participant}.bin'
        model.dump_model(txt_model_path)
        with open(bin_model_path, 'wb+') as w:
            w.write(model.save_raw()[4:])
        X.to_csv(f'{model_dir}/{participant}_X.csv')
        Y.to_csv(f'{model_dir}/{participant}_Y.csv')

    if average:
        params, scores = np.average(test_results, axis=0)
        return params, scores
    else:
        return test_results


# train and get model
def participant_train_for_model(participant, train_dir, tune_parameters=False):
    all_models = []
    all_test_features = []
    all_test_scores = []
    conf_mtx = np.zeros((2, 2))  # 2 X 2 confusion matrix

    test_numbers = []
    for test_number in os.listdir(f'{settings.baseline_test_dir}/{settings.participants[0]}'):
        if test_number.endswith('.csv') and test_number[:-4].isdigit():
            test_numbers += [int(test_number[:-4])]
    test_numbers.sort()

    for test_number in test_numbers:
        ts, test_features, test_labels = load_dataset(directory=f'{settings.baseline_test_dir}/{participant}', filename=f'{test_number}.csv', selected_column_names=settings.basic_selected_features, screen_out_timestamps=None)
        if os.path.exists(f'{train_dir}/{participant}.csv'):
            _, train_features, train_labels = load_dataset(
                directory=train_dir,
                filename=f'{participant}.csv',
                selected_column_names=settings.basic_selected_features,
                screen_out_timestamps=ts
            )
        else:
            _, train_features, train_labels = load_dataset(
                directory=f'{train_dir}/{test_number}',
                filename=f'{participant}.csv',
                selected_column_names=settings.basic_selected_features,
                screen_out_timestamps=ts
            )

        # configure test dataset
        scaler = MinMaxScaler()
        scaler.fit(test_features)
        test_features_scaled = scaler.transform(test_features)
        test_features = pd.DataFrame(test_features_scaled, index=test_features.index, columns=test_features.columns)

        k_folds = []
        splitter = StratifiedKFold(n_splits=5, shuffle=True)
        for idx, (train_indices, test_indices) in enumerate(splitter.split(train_features, train_labels)):
            try:
                x_train = train_features.iloc[train_indices]
                y_train = train_labels.iloc[train_indices]
                x_test = train_features.iloc[test_indices]
                y_test = train_labels.iloc[test_indices]
                k_folds.append((x_train, y_train, x_test, y_test))
            except ValueError:
                test = False
                continue

        k_folds_sampled = []
        for idx, (x_train, y_train, x_test, y_test) in enumerate(k_folds):
            try:
                sampler = SMOTE()
                x_sample, y_sample = sampler.fit_resample(x_train, y_train)
                x_sample = pd.DataFrame(x_sample, columns=x_train.columns)
                y_sample = pd.Series(y_sample)
                k_folds_sampled.append((x_sample, y_sample, x_test, y_test))
            except ValueError:
                test = False
                continue

        k_folds_scaled = []
        for x_train, y_train, x_test, y_test in k_folds_sampled:
            scaler = MinMaxScaler()
            scaler.fit(x_train)
            x_train_scale = scaler.transform(x_train)
            x_test_scale = scaler.transform(x_test)
            x_train = pd.DataFrame(x_train_scale, index=x_train.index, columns=x_train.columns)
            x_test = pd.DataFrame(x_test_scale, index=x_test.index, columns=x_test.columns)
            k_folds_scaled.append((x_train, y_train, x_test, y_test))

        train_data = xgb.DMatrix(data=train_features, label=train_labels.to_numpy())

        # Parameter tuning / grid search
        train_parameters = {
            'max_depth': 6,
            'min_child_weight': 1,
            'eta': .3,
            'subsample': 1,
            'colsample_bytree': 1,
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'verbosity': 0,
            'eval_metric': "auc"
        }
        if tune_parameters:
            temporary_parameters = {
                'max_depth': 6,
                'min_child_weight': 1,
                'eta': .3,
                'subsample': 1,
                'colsample_bytree': 1,
                'objective': 'binary:logistic',
                'booster': 'gbtree',
                'verbosity': 0,
                'eval_metric': "auc"
            }
            grid_search_params = [(max_depth, min_child_weight) for max_depth in range(0, 12) for min_child_weight in range(0, 8)]
            current_test_auc = -float("Inf")
            for max_depth, min_child_weight in grid_search_params:
                try:
                    # Update our parameters
                    temporary_parameters['max_depth'] = max_depth
                    temporary_parameters['min_child_weight'] = min_child_weight
                    # Run CV
                    cv_results = xgb.cv(temporary_parameters, train_data, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                    # Update best MAE
                    mean_mae = cv_results['test-auc-mean'].max()
                    if mean_mae > current_test_auc:
                        current_test_auc = mean_mae
                        train_parameters['max_depth'] = max_depth
                        train_parameters['min_child_weight'] = min_child_weight
                except xgb.core.XGBoostError:
                    continue

            grid_search_params = [(subsample, colsample) for subsample in [i / 10. for i in range(7, 11)] for colsample in [i / 10. for i in range(7, 11)]]
            current_test_auc = -float("Inf")
            temporary_parameters = {'subsample': None, 'colsample_bytree': None}
            # We start by the largest values and go down to the smallest
            for sub_sample, col_sample in reversed(grid_search_params):
                try:
                    # We update our parameters
                    train_parameters['subsample'] = sub_sample
                    train_parameters['colsample_bytree'] = col_sample
                    # Run CV
                    cv_results = xgb.cv(train_parameters, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                    mean_mae = cv_results['test-auc-mean'].max()
                    if mean_mae > current_test_auc:
                        current_test_auc = mean_mae
                        temporary_parameters = {'subsample': sub_sample, 'colsample_bytree': col_sample}
                except xgb.core.XGBoostError:
                    continue
            train_parameters['subsample'] = temporary_parameters['subsample']
            train_parameters['colsample_bytree'] = temporary_parameters['colsample_bytree']

            current_test_auc = -float("Inf")
            temporary_parameters = None
            for eta in [.3, .2, .1, .05, .01, .005]:
                try:
                    # We update our parameters
                    train_parameters['eta'] = eta
                    # Run and time CV
                    cv_results = xgb.cv(train_parameters, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                    # Update best score
                    mean_mae = cv_results['test-auc-mean'].max()
                    if mean_mae > current_test_auc:
                        current_test_auc = mean_mae
                        temporary_parameters = eta
                except xgb.core.XGBoostError:
                    continue
            train_parameters['eta'] = temporary_parameters

            current_test_auc = -float("Inf")
            temporary_parameters = None
            gamma_range = [i / 10.0 for i in range(0, 25)]
            for gamma in gamma_range:
                try:
                    # We update our parameters
                    train_parameters['gamma'] = gamma
                    # Run and time CV
                    cv_results = xgb.cv(train_parameters, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                    # Update best score
                    mean_mae = cv_results['test-auc-mean'].max()
                    if mean_mae > current_test_auc:
                        current_test_auc = mean_mae
                        temporary_parameters = gamma
                except xgb.core.XGBoostError:
                    continue
            train_parameters['gamma'] = temporary_parameters

        for x_train, y_train, x_test, y_test in k_folds_scaled:
            train_data = xgb.DMatrix(data=x_train, label=y_train.to_numpy())
            evaluation_data = xgb.DMatrix(data=x_test, label=y_test.to_numpy())
            test_data = xgb.DMatrix(data=test_features, label=test_labels.to_numpy())

            try:
                results = {}
                booster = xgb.train(train_parameters, dtrain=train_data, num_boost_round=1000, early_stopping_rounds=25, evals=[(evaluation_data, 'test')], verbose_eval=False, evals_result=results)

                predicted_probabilities = booster.predict(data=test_data, ntree_limit=booster.best_ntree_limit)
                predicted_labels = np.where(predicted_probabilities > 0.5, 1, 0)

                # acc = balanced_accuracy_score(test_labels, predicted_labels)
                f1 = f1_score(test_labels, predicted_labels, average='macro')
                roc_auc = roc_auc_score(test_labels, predicted_probabilities)
                # tpr = recall_score(test_labels, predicted_labels)
                # tnr = recall_score(test_labels, predicted_labels, pos_label=0)

                all_models += [booster]
                all_test_features += [test_features]
                all_test_scores += [(f1, roc_auc)]
                conf_mtx += confusion_matrix(test_labels, predicted_labels)

            except xgb.core.XGBoostError:
                print(f'(test dataset #{test_number} error)', end=' ', flush=True)
                continue

    return all_models, all_test_features, all_test_scores, conf_mtx


# save results in a file
def write_results(dir, scores, params, scores_cols, params_cols):
    with open(f"{dir}/scores.csv", "w+") as w_scores, open(f"{settings.no_filter_test_results_dir}/params.csv", "w+") as w_params:
        w_params.write('Participant,{}\n'.format(','.join(params_cols)))
        w_scores.write('Participant,{}\n'.format(','.join(scores_cols)))
        for participant in params:
            w_params.write(participant)
            w_scores.write(participant)
            for param_col in params_cols:
                w_params.write(',{}'.format(params[participant][param_col]))
            for score_col in scores_cols:
                w_scores.write(',{}'.format(scores[participant][score_col]))
            w_params.write('\n')
            w_scores.write('\n')


# trains and tests on test datasets with holdout
def participant_train_test_xgboost_holdout():
    pass


# calculates average of numbers in array
def average(values):
    return sum(values) / len(values)
