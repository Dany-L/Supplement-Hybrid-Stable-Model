# Hybrid Stable Model that interconnect linear system with robust RNN

This repository is forked from https://github.com/AlexandraBaier/Supplement_Physics_Residual-Bounded_LSTM.git

Clone this repository and install dependencies:
```shell
git clone https://github.com/AlexandraBaier/Supplement_Physics_Residual-Bounded_LSTM.git
cd Supplement_Physics_Residual-Bounded_LSTM
pip install .
```

All directories and files will be created within the cloned directory.

Run the following to download all datasets and set up the required directories:
```shell
python scripts/setup_environment.py
```

To run the experiments for the some toy datasets run the following two scripts in order:
```shell
python scripts/run_experiment_toy_dataset.py {system} {device}
```
where `device` is the identifier (an integer starting at 0) for the GPU to run the experiments on. 
If you only have one GPU, set the value to `0` and `system`can be `cartpole`, `mass-spring-damper`or `pendulum`.

If these scripts are stopped for any reason, you can rerun them without issue. 
`run_experiment_toy_dataset.py` remembers what models where already trained and validated.

Trained models will be stored in the system directory that are also contains the dataset under `results`.
The models will be placed in `models`, the environment will be stored in an `{system}.env` file and the progress of the session will be stored in the `configuration` folder.

To summarize the results on the test set run the following script:
```shell
python scripts/summarize_results.py
```

To summarize the results for the ablation study on the clipping-accuracy trade-off run the following:
```shell
python scripts/summarize_threshold_results.py
```

Both `summarize` scripts will deposit CSV files in the results directories of the respective dataset.
