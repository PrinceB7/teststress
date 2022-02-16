from scripts import settings
from scripts import utils

ts_ranges = {
    # 'aliceblackwood123@gmail.com': (1587826800000, 1587913200000),
    # 'azizsambo58@gmail.com': (1587740400000, 1587826800000),  # change
    # 'jskim@nsl.inha.ac.kr': (1587826800000, 1587913200000),
    # 'jumabek4044@gmail.com': (1587999600000, 1588086000000),  # change
    # 'laurentkalpers3@gmail.com': (1587826800000, 1587913200000),
    # 'mr.khikmatillo@gmail.com': (1587826800000, 1587913200000),
    'nazarov7mu@gmail.com': (1587826800000, 1587913200000),  # change
    # 'nnarziev@gmail.com': (1587654000000, 1587740400000),  # change
    # 'nslabinha@gmail.com': (1587740400000, 1587826800000),  # change
    # 'salman@nsl.inha.ac.kr': (1587913200000, 1587999600000),  # change
}


def load_dataset(fp, ts_range, cols_count):
    data = []
    read_ts = set()
    for line in fp.readlines()[1:]:
        cells = line.split(',')
        timestamp = int(cells[0])
        if timestamp not in read_ts and ts_range[0] < timestamp < ts_range[1]:
            if cols_count == 1:
                data += [(timestamp, int(cells[1]))]
                read_ts.add(timestamp)
            elif cols_count == 3:
                data += [(timestamp, float(cells[1]), float(cells[2]), float(cells[3]))]
                read_ts.add(timestamp)
    data.sort(key=lambda x: x[0])
    delay_sum = 0
    for i in range(1, len(data)):
        delay_sum += data[i][0] - data[i - 1][0]
    return data, delay_sum / (len(data) - 1)


for participant in ts_ranges.keys():
    print(participant)

    r_acc = open(f'{settings.raw_dataset_dir}/{participant}/ACCELEROMETER.csv', 'r')
    r_ppg = open(f'{settings.raw_dataset_dir}/{participant}/LIGHT_INTENSITY.csv', 'r')
    w = open(f'{settings.one_day_dataset_dir}/{participant}.csv', 'w+')

    ts_range = ts_ranges[participant]
    acc_dataset, acc_delay = load_dataset(fp=r_acc, ts_range=ts_range, cols_count=3)
    ppg_dataset, ppg_delay = load_dataset(fp=r_ppg, ts_range=ts_range, cols_count=1)

    w.write("Timestamp,ACC magnitude,PPG stdev\n")

    ts = ts_range[0]
    while ts < ts_range[1]:
        acc_selected = utils.select_data(dataset=acc_dataset, from_ts=ts, till_ts=ts + 60000, with_timestamp=True)
        ppg_selected = utils.select_data(dataset=ppg_dataset, from_ts=ts, till_ts=ts + 60000)
        if acc_selected is not None:
            acc_magnitude = utils.average(values=utils.get_activeness_scores(acc_values=acc_selected))
        else:
            acc_magnitude = '=na()'
        if ppg_selected is not None:
            ppg_stdev = utils.get_ppg_stdev(ppg_values=ppg_selected)
        else:
            ppg_stdev = '=na()'
        ts += 60000
        w.write(f'{ts},{acc_magnitude},{ppg_stdev}\n')

    r_acc.close()
    r_ppg.close()
    w.close()
