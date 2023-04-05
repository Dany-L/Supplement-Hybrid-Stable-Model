import argparse
import pathlib

from deepsysid.cli.download import TOY_DATASET_FOLDERNAMES_DICT, TOY_DATASET_ZIP_BASE_NAME

from pbrl.utils import load_environment, run_full_gridsearch_session


def run_system_experiment(path: pathlib.Path, system_name: str)-> None:
    report_path = path.joinpath('configuration').joinpath(f'progress-{system_name}.json')
    environment_path = path.joinpath(f'{system_name}.env')

    environment = load_environment(environment_path)

    run_full_gridsearch_session(
        report_path=report_path,
        environment=environment
    )

def main():
    parser = argparse.ArgumentParser('Run experiments for toy dataset.')
    parser.add_argument('system_name')
    args = parser.parse_args()

    system_name = str(args.system_name)

    main_path = pathlib.Path(__file__).parent.parent.absolute().joinpath(TOY_DATASET_ZIP_BASE_NAME)

    run_system_experiment(
        path=main_path.joinpath(system_name),
        system_name=system_name,
    )

if __name__ == '__main__':
    main()
