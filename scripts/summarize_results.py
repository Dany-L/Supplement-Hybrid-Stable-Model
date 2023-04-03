import pathlib
from typing import Set, List

import h5py
import numpy as np
import pandas as pd
from deepsysid.pipeline.configuration import ExperimentGridSearchTemplate, ExperimentConfiguration
from deepsysid.pipeline.data_io import build_score_file_name, build_explanation_result_file_name
from deepsysid.pipeline.evaluation import ReadableEvaluationScores
from deepsysid.pipeline.gridsearch import ExperimentSessionReport
from deepsysid.cli.download import TOY_DATASET_FOLDERNAMES_DICT, TOY_DATASET_ZIP_BASE_NAME


def get_best_models(report_path: pathlib.Path) -> Set[str]:
    report = ExperimentSessionReport.parse_file(report_path)

    if report.best_per_class is None or report.best_per_base_name is None:
        raise ValueError(
            'Did not run "deepsysid session" with action=TEST_BEST yet. '
            'Best performing models have not been identified yet.'
        )

    best_models = set(report.best_per_class.values()).union(report.best_per_base_name.values())

    return best_models


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


def summarize_experiment(
    report_path: pathlib.Path,
    configuration_path: pathlib.Path,
    result_directory: pathlib.Path,
    horizons: List[int]
) -> None:
    best_models = get_best_models(report_path)

    configuration = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_file(configuration_path)
    )
    n_runs = configuration.session.total_runs_for_best_models

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

    for data_rel_dict in TOY_DATASET_FOLDERNAMES_DICT.values():
        system_name = pathlib.Path(data_rel_dict).parent
        system_root_folder = main_path.joinpath(TOY_DATASET_FOLDERNAMES_DICT, system_name)
        summarize_experiment(
            report_path=system_root_folder.joinpath('configuration', f'progress-{system_root_folder}.json'),
            configuration_path=main_path.joinpath('configuration', f'{system_name}.json'),
            result_directory=system_root_folder.joinpath('results'),
            horizons=[10, 100, 400]
        )


if __name__ == '__main__':
    main()
