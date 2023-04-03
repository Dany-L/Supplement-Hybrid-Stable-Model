import dataclasses
import pathlib
import subprocess
from typing import List

from deepsysid.cli.download import TOY_DATASET_FOLDERNAMES_DICT, TOY_DATASET_ZIP_BASE_NAME


@dataclasses.dataclass
class Directories:
    datasets: pathlib.Path
    models: pathlib.Path
    results: pathlib.Path
    configuration: pathlib.Path
    name: str


def create_directories(main_path: pathlib.Path) -> List[Directories]:
    dirs = list()
    for dataset_rel_dir in TOY_DATASET_FOLDERNAMES_DICT.values():
        name = str(pathlib.Path(dataset_rel_dir).parent)
        system_root_path = main_path.joinpath(TOY_DATASET_ZIP_BASE_NAME, pathlib.Path(dataset_rel_dir).parent)
        system_root_path.joinpath('models').mkdir(exist_ok=True)
        system_root_path.joinpath('configuration').mkdir(exist_ok=True)
        system_root_path.joinpath('results').mkdir(exist_ok=True)

        dirs.append(
            Directories(
                models=system_root_path.joinpath('models'),
                results=system_root_path.joinpath('results'),
                configuration=system_root_path.joinpath('configuration'),
                datasets=main_path.joinpath(TOY_DATASET_ZIP_BASE_NAME, dataset_rel_dir),
                name = name
            )
        )
    return dirs


def create_environment(dirs: List[Directories], main_path: pathlib.Path) -> None:

    for system in dirs:
        env_file = system.datasets.parent.joinpath(f'{system.name}.env')
        with env_file.open(mode='w') as f:
            f.write('\n'.join([
                f'DATASET_DIRECTORY={system.datasets}',
                f'MODELS_DIRECTORY={system.models}',
                f'RESULT_DIRECTORY={system.results}',
                f'CONFIGURATION={main_path.joinpath("configuration", f"{system.name}.json")}'
            ]))


def download_dataset(target_directory: pathlib.Path) -> None:
    subprocess.call([
        'deepsysid',
        'download',
        'toy_dataset',
        target_directory
    ])


def main():
    main_path = pathlib.Path(__file__).parent.parent.absolute()
    download_dataset(main_path)
    dirs = create_directories(main_path)
    create_environment(dirs, main_path)
    


if __name__ == '__main__':
    main()
