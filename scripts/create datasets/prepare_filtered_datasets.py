from scripts import settings
import shutil
import os


def main():
    # ppg filter
    for test_number in settings.tests_ppg_thresholds:
        if not os.path.exists(path=f'/Users/kevin/Desktop/stress/data/processed dataset/2. ppg filtered dataset/{test_number}'):
            os.mkdir(f'/Users/kevin/Desktop/stress/data/processed dataset/2. ppg filtered dataset/{test_number}')
        for participant in settings.tests_ppg_thresholds[test_number]:
            shutil.copyfile(
                src=f'/Users/kevin/Desktop/stress-gridsearch/data/ppg/filter thresholds/{settings.tests_ppg_thresholds[test_number][participant]}/{participant}.csv',
                dst=f'/Users/kevin/Desktop/stress/data/processed dataset/2. ppg filtered dataset/{test_number}/{participant}.csv'
            )
    # acc filter
    for test_number in settings.tests_acc_thresholds:
        if not os.path.exists(path=f'/Users/kevin/Desktop/stress/data/processed dataset/3. acc filtered dataset/{test_number}'):
            os.mkdir(f'/Users/kevin/Desktop/stress/data/processed dataset/3. acc filtered dataset/{test_number}')
        for participant in settings.tests_acc_thresholds[test_number]:
            shutil.copyfile(
                src=f'/Users/kevin/Desktop/stress-gridsearch/data/acc/filter thresholds/{settings.tests_acc_thresholds[test_number][participant]}/{participant}.csv',
                dst=f'/Users/kevin/Desktop/stress/data/processed dataset/3. acc filtered dataset/{test_number}/{participant}.csv'
            )
    # combined filter
    for test_number in settings.tests_combined_thresholds['PPG']:
        if not os.path.exists(path=f'/Users/kevin/Desktop/stress/data/processed dataset/4. combined filtered dataset/{test_number}'):
            os.mkdir(f'/Users/kevin/Desktop/stress/data/processed dataset/4. combined filtered dataset/{test_number}')
        for participant in settings.tests_combined_thresholds['PPG'][test_number]:
            shutil.copyfile(
                src=f'/Users/kevin/Desktop/stress-gridsearch/data/combined/filter thresholds/{settings.tests_combined_thresholds["PPG"][test_number][participant]}/{settings.tests_combined_thresholds["ACC"][test_number][participant]}/{participant}.csv',
                dst=f'/Users/kevin/Desktop/stress/data/processed dataset/4. combined filtered dataset/{test_number}/{participant}.csv'
            )


if __name__ == '__main__':
    main()
