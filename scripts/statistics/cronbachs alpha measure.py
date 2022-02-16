from scripts import settings
from scripts import utils
import pingouin as pg
import pandas as pd


def main():
    for participant in settings.participants:
        scores_untouched = []
        for gt in utils.load_ground_truths(participant=participant):
            lose_control = gt[2]
            difficult = gt[3]
            confident = gt[4]
            your_way = gt[5]
            likert_stress_level = gt[6]
            pss_score = (lose_control + difficult + (6 - confident) + (6 - your_way)) / 4
            scores_untouched += [[likert_stress_level, pss_score]]
        print(f'{participant}\t{pg.cronbach_alpha(data=pd.DataFrame(scores_untouched, columns=list("xy")))}\t', end='')

        scores_filtered = []
        for gt in utils.load_unfiltered_ground_truths(participant=participant):
            lose_control = gt[2]
            difficult = gt[3]
            confident = gt[4]
            your_way = gt[5]
            likert_stress_level = gt[6]
            pss_score = (lose_control + difficult + (6 - confident) + (6 - your_way)) / 4
            scores_filtered += [[likert_stress_level, pss_score]]
        print(f'{pg.cronbach_alpha(data=pd.DataFrame(scores_filtered, columns=list("xy")))}')


if __name__ == '__main__':
    main()
