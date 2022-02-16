import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scripts import settings
from scripts import utils
import pandas as pd
import numpy as np
import shap
import os


def save_fig_as_plot(filter_name, title, participant, figure):
    if not os.path.exists(f'{settings.plots_dir}/{filter_name}'):
        os.mkdir(f'{settings.plots_dir}/{filter_name}')
        os.mkdir(f'{settings.plots_dir}/{filter_name}/{participant}')
    elif not os.path.exists(f'{settings.plots_dir}/{filter_name}/{participant}'):
        os.mkdir(f'{settings.plots_dir}/{filter_name}/{participant}')

    if title == 'shap':
        figure.savefig(f'{settings.plots_dir}/{filter_name}/{participant}/{title}.svg', dpi=200, format="svg")
    else:
        figure.write_image(f'{settings.plots_dir}/{filter_name}/{participant}/{title}.png', scale=2.0)


def plot_test_graphs(filter_name, participants):
    # region detect filter
    train_dir = None
    test_numbers = None
    if filter_name == 'combined':
        train_dir = settings.combined_filtered_dataset_dir
        test_numbers = settings.combined_best_tests
    elif filter_name == 'ppg':
        train_dir = settings.ppg_filtered_dataset_dir
        test_numbers = settings.ppg_best_tests
    elif filter_name == 'acc':
        train_dir = settings.acc_filtered_dataset_dir
        test_numbers = settings.acc_best_tests
    elif filter_name == 'no filter':
        train_dir = settings.not_filtered_dataset_dir
        test_numbers = settings.no_filter_best_tests

    if None in [train_dir, test_numbers]:
        return
    # endregion

    for participant in participants:
        models, X_tests, conf_mtx = utils.participant_train_for_model(participant=participant, train_dir=train_dir)

        # region plot confusion matrix
        fig = go.Figure(
            ff.create_annotated_heatmap(
                x=['Prediction = no stress', 'Prediction = stress'],
                y=['Ground truth = stress', 'GT = no stress'],
                z=np.flip(conf_mtx, axis=0),
                colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']],
            ), layout=go.Layout(
                paper_bgcolor='#f2f8ff',
                plot_bgcolor='#f2f8ff'
            )
        )
        fig.update_layout(
            title_text=f'{filter_name}, {participant}',
            xaxis_title_text='Prediction',
            yaxis_title_text='GT',
            bargap=1,
        )
        # fig.show()
        save_fig_as_plot(filter_name=filter_name, title='confusion_matrix', participant=participant, figure=fig)
        # endregion

        # region plot feature importance
        feature_importances = {}
        for model in models:
            imp = model.get_fscore()
            for k, v in imp.items():
                if k in feature_importances:
                    feature_importances[k] += v
                else:
                    feature_importances[k] = v
        feature_importances = [(k, float(v) / 5) for k, v in feature_importances.items()]
        df = pd.DataFrame(feature_importances, columns=['name', 'importance']).sort_values('importance', ascending=False).head(50)
        order = {}
        for name, index in zip(list(df.loc[:, 'name']), list(df.loc[:, 'name'].index)):
            order[name] = index
        fig = go.Figure(
            go.Bar(
                x=df.loc[:, 'name'][[order['mean_nni'], order['sdnn'], order['rmssd'], order['nni_50'], order['lf'], order['hf'], order['lf_hf_ratio'], order['ratio_sd2_sd1'], order['sd2'], order['sampen']]],
                y=df.loc[:, 'importance'][[order['mean_nni'], order['sdnn'], order['rmssd'], order['nni_50'], order['lf'], order['hf'], order['lf_hf_ratio'], order['ratio_sd2_sd1'], order['sd2'], order['sampen']]],
                marker=dict(color='#444')
            ), go.Layout(
                paper_bgcolor='#f2f8ff',
                plot_bgcolor='#f2f8ff'
            )
        )
        fig.update_layout(
            title_text=f'{filter_name}, {participant}',
            yaxis_title_text='Feature importances'
        )
        # fig.show()
        save_fig_as_plot(filter_name=filter_name, title='feature_importance', participant=participant, figure=fig)
        # endregion

        # region plot shap
        shap_values = []
        test_features = []
        for model, X_test in zip(models, X_tests):
            explainer = shap.TreeExplainer(model)
            shap_value = explainer.shap_values(X_test, tree_limit=model.best_ntree_limit)
            shap_values.append(shap_value)
            test_features.append(X_test)
        shap_values = np.vstack(shap_values)
        test_features = pd.concat(test_features, axis=0)
        fig = plt.figure()
        fig.suptitle(f'{filter_name}, {participant}', fontsize=20)
        plt.rcParams['axes.facecolor'] = '#f2f8ff'
        plt.xlim(-6, 6)
        shap.summary_plot(shap_values, test_features, show=False, cmap=plt.get_cmap('gray'), sort=False)
        # plt.show()
        save_fig_as_plot(filter_name=filter_name, title='shap', participant=participant, figure=fig)
        # endregion


def main():
    participants = ['azizsambo58@gmail.com', 'nslabinha@gmail.com', 'nnarziev@gmail.com', 'jskim@nsl.inha.ac.kr']
    plot_test_graphs(filter_name='no filter', participants=participants)
    # plot_test_graphs(filter_name='combined', participants=participants)


if __name__ == '__main__':
    main()

# explainer = shap.TreeExplainer(model=model)
# shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
