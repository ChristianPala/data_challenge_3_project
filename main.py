# Libraries
from preprocessing.pre_main import main as prepro
from feature_engineering.fe_main import main as fe
from feature_selection.fs_main import main as fs
from dimensionality_reduction.dr_main import main as dr
from error_analysis.error_analysis_with_shap_values import main as ea
from modelling.model_main import main as run_models
import os
from pathlib import Path


def create_directories():
    # data dir
    data_path = Path('data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    params_path = Path('data', 'best_params')
    if not os.path.exists(params_path):
        os.makedirs(params_path)


def main(run_error_analysis=False, run_model=False):
    create_directories()
    prepro(preprocessing_only=True)
    fe()
    fs(run_wrappers=False, run_correlation=False)
    dr(run_full=False, run_dsa=False)

    if run_model:
        run_models()

    if run_error_analysis:
        ea()


if __name__ == '__main__':
    main(run_error_analysis=False, run_model=False)
