import pathlib
from typing import Set, List, Union
import argparse
import re
import time

import h5py
import numpy as np
import pandas as pd
from deepsysid.pipeline.configuration import ExperimentGridSearchTemplate, ExperimentConfiguration
from deepsysid.pipeline.data_io import build_score_file_name, build_explanation_result_file_name, build_result_file_name
from deepsysid.pipeline.evaluation import ReadableEvaluationScores
from deepsysid.pipeline.gridsearch import ExperimentSessionReport
from deepsysid.pipeline.testing.base import TestSequenceResult
from deepsysid.cli.download import TOY_DATASET_FOLDERNAMES_DICT, TOY_DATASET_ZIP_BASE_NAME
import matplotlib.pyplot as plt
import tikzplotlib

from pbrl.utils import save_pyplot

plt.rcParams['text.usetex'] = True


def get_best_models(report_path: pathlib.Path) -> Set[str]:
    report = ExperimentSessionReport.parse_file(report_path)

    if report.best_per_class is None or report.best_per_base_name is None:
        raise ValueError(
            'Did not run "deepsysid session" with action=TEST_BEST yet. '
            'Best performing models have not been identified yet.'
        )

    best_models = set(report.best_per_class.values()).union(report.best_per_base_name.values())

    return best_models

def get_all_models(report_path: pathlib.Path) -> Set[str]:
    report = ExperimentSessionReport.parse_file(report_path)

    all_models = set(report.validated_models)

    return all_models


def summarize_prediction_scores(
    configuration: ExperimentConfiguration,
    models: Set[str],
    result_directory: pathlib.Path,
    horizons: List[int]
) -> pd.DataFrame:
    rows = []
    for model in models:
        score_file_name = build_score_file_name(
            mode='test',
            window_size=configuration.window_size,
            horizon_size=configuration.horizon_size,
            extension='json'
        )
        scores = ReadableEvaluationScores.parse_file(
            result_directory.joinpath(model).joinpath(score_file_name)
        )
        row = [model]
        for horizon in horizons:
            nrmse_multi = np.mean(
                scores.scores_per_horizon[horizon]['nrmse']
            )
            row.append(nrmse_multi)
        rows.append(row)

    df = pd.DataFrame(
        data=rows,
        columns=['model'] + [f'H={horizon}' for horizon in horizons]
    )
    return df

def print_comparison_plot(
    configuration: ExperimentConfiguration,
    models: Set[str],
    result_directory: pathlib.Path,
    sample_number: int
) -> None:

    num_inputs = len(configuration.control_names)
    num_outputs = len(configuration.state_names)

    t = np.linspace(
        0, 
        (configuration.horizon_size-1)*configuration.time_delta,
        configuration.horizon_size
    )

    # input identical across all models
    result_file_name = build_result_file_name(
        'test',
        configuration.window_size,
        configuration.horizon_size,
        'hdf5'
    )
    results = h5py.File(
        result_directory.joinpath(list(models)[0]).joinpath(result_file_name),
        'r'
    )
    fig, axs = plt.subplots(
        nrows=num_inputs,
        ncols=1,
        tight_layout = True,
        squeeze=False
    )
    fig.suptitle('$Input$')
    for element, ax in zip(range(num_inputs), axs[:,0]):
        ax.plot(t, results['main'][f'{sample_number}']['inputs']['control'][:,element])
        ax.set_title(f'${configuration.control_names[element]}$')
        ax.grid()
    save_pyplot(
        'input',
        result_directory,
        sample_number
    )

    fig, axs = plt.subplots(
        nrows=num_inputs,
        ncols=1,
        tight_layout = True,
        squeeze=False
    )
    fig.suptitle('Outputs')
    for idx, model in enumerate(models):
        result_file_name = build_result_file_name(
            'test',
            configuration.window_size,
            configuration.horizon_size,
            'hdf5'
        )
        results = h5py.File(
            result_directory.joinpath(model).joinpath(result_file_name),
            'r'
        )
        if idx==0:
            for element, ax in zip(range(num_outputs), axs[:,0]):
                (line,) = ax.plot(
                    t, 
                    results['main'][f'{sample_number}']['outputs']['true_state'][:,element])
                line.set_label(f'true')
                line.set_linestyle('dashed')
                ax.set_title(f'${configuration.state_names[element]}$')
                ax.grid()

        for element, ax in zip(range(num_outputs), axs[:,0]):
            (line,) = ax.plot(
                t, 
                results['main'][f'{sample_number}']['outputs']['pred_state'][:,element]
            )
            line.set_label(f'{model}')
            ax.legend()
   
    save_pyplot(
            'output',
            result_directory,
            sample_number
        )
    
def print_training_trajectories(
    trajectory_names: List[str],
    models: Set[str],
    model_directory: pathlib.Path,
) -> None:

    for trajectory_name in trajectory_names:

        fig, ax = plt.subplots()
        fig.suptitle(trajectory_name)

        for model in models:
            metadata_file_name = f'{model}-metadata.hdf5'
            metadata = h5py.File(
                model_directory.joinpath(model).joinpath(metadata_file_name),
                mode='r'
            )
            y = [0]
            try:
                y_raw = np.array(metadata[trajectory_name])
                if len(y_raw.shape) > 1:
                    y = y_raw[:,1]
                else:
                    y = y_raw
            except:
                print(f'{trajectory_name} does not exist for {model}.')
            
            (line,) = ax.plot(np.linspace(0, len(y)-1, len(y)), y)
            line.set_label(f'{model}')
            ax.grid()
            ax.legend()
        save_pyplot(
            trajectory_name,
            model_directory,
            0
        )


def print_training_scalars(
    scalar_names: List[str],
    models: Set[str],
    model_directory: pathlib.Path,
) -> None:
    time_regexp = re.compile('^.*_time_.*')
    rows = list()
    for model in models:
        row = [model]
        for scalar_name in scalar_names:
            
            metadata_file_name = f'{model}-metadata.hdf5'
            metadata = h5py.File(
                model_directory.joinpath(model).joinpath(metadata_file_name),
                mode='r'
            )

            try:
                value = np.squeeze(np.array(metadata[scalar_name]))
                if time_regexp.match(scalar_name):
                    time_struct = time.gmtime(float(value))
                    row.append(time.strftime('%H:%M:%S', time_struct))
                else:
                    row.append(value)
            except:
                row.append('-')
        rows.append(row)
    df = pd.DataFrame(
        data=rows,
        columns=['model'] + [scalar_name for scalar_name in scalar_names]
    )
    df.to_csv(
        path_or_buf=model_directory.joinpath('metadata.csv')
    )


def summarize_experiment(
    report_path: pathlib.Path,
    configuration_path: pathlib.Path,
    result_directory: pathlib.Path,
    model_directory: pathlib.Path,
    horizons: List[int]
) -> None:
    best_models = get_best_models(report_path)
    all_validated_models = get_all_models(report_path)

    configuration = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_file(configuration_path)
    )
    n_runs = configuration.session.total_runs_for_best_models

    print_comparison_plot(
        configuration,
        best_models,
        result_directory,
        0
    )

    print_training_trajectories(
        configuration.session.training_trajectory,
        best_models,
        model_directory
    )

    print_training_scalars(
        configuration.session.training_scalar,
        all_validated_models,
        model_directory
    )

    prediction_scores = summarize_prediction_scores(
        configuration,
        best_models,
        result_directory,
        horizons
    )
    prediction_scores['run'] = 0
    for run_idx in range(1, n_runs):
        additional_prediction_scores = summarize_prediction_scores(
            configuration,
            best_models,
            result_directory=result_directory.joinpath(f'repeat-{run_idx}'),
            horizons=horizons
        )
        additional_prediction_scores['run'] = run_idx
        prediction_scores = pd.concat((prediction_scores, additional_prediction_scores))

    # https://stackoverflow.com/a/53522680
    stats = prediction_scores\
        .groupby(['model'])[[f'H={horizon}' for horizon in horizons]]\
        .agg(['mean', 'count', 'std'])
    for horizon in horizons:
        mean = stats[(f'H={horizon}', 'mean')]
        count = stats[(f'H={horizon}', 'count')]
        std = stats[(f'H={horizon}', 'std')]
        stats[(f'H={horizon}', 'ci95-width')] = 1.96 * std / np.sqrt(count)
        stats[(f'H={horizon}', 'ci95-lo')] = mean - stats[(f'H={horizon}', 'ci95-width')]
        stats[(f'H={horizon}', 'ci95-hi')] = mean + stats[(f'H={horizon}', 'ci95-width')]

    prediction_scores.to_csv(
        result_directory.joinpath('summary-prediction.csv'),
    )
    stats.to_csv(
        result_directory.joinpath('summary-prediction-ci.csv'),
    )


def main():
    main_path = pathlib.Path(__file__).parent.parent.absolute()

    parser = argparse.ArgumentParser('Run experiments for toy dataset.')
    parser.add_argument('system_name')
    args = parser.parse_args()

    system_name = str(args.system_name)
    system_root_folder = main_path.joinpath(TOY_DATASET_ZIP_BASE_NAME, system_name)

    summarize_experiment(
        report_path=system_root_folder.joinpath('configuration', f'progress-{system_name}.json'),
        configuration_path=main_path.joinpath('configuration', f'{system_name}.json'),
        result_directory=system_root_folder.joinpath('results'),
        model_directory=system_root_folder.joinpath('models'),
        horizons=[10, 100, 400]
    )

if __name__ == '__main__':
    main()
