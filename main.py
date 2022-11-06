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


def main(preprocessing_only=True,
         run_error_analysis=False,
         run_model=False,
         run_wrap_methods=False,
         run_dr_on_full_dataset=False,
         run_data_scarcity_analysis=False) -> None:
    """
    Method to run the entire project, it automatically downloads nltk dependencies for NLP part.
    @param preprocessing_only: run only the preprocessing phase, excluding churn analysis and exploratory data analysis
    @param run_error_analysis: either to run the error analysis done with shap
    @param run_model: run the models
    @param run_wrap_methods: run the wrapper feature selection methods
    @param run_dr_on_full_dataset: run dimensionality reduction also on the full dataset
    (not only on the one outputted by feature selection methods)
    @param run_data_scarcity_analysis: run the data scarcity analysis when performing dimensionality reduction
    @return: None
    """

    print('\n> creating missing directories')
    create_directories()
    file_path = Path('data', 'online_sales_dataset.xlsx')
    if not os.path.exists(file_path):
        print('Please put the "online_sales_dataset.xlsx" file into the "data" folder')
        exit(0)
    print('\n*** PREPROCESSING PHASE ***')
    prepro(preprocessing_only=preprocessing_only)
    print('\n*** FEATURE ENGINEERING PHASE ***')
    fe()
    print('\n*** FEATURE SELECTION PHASE ***')
    fs(run_wrappers=run_wrap_methods, run_correlation=False)
    print('\n*** DIMENSIONALITY REDUCTION PHASE ***')
    dr(run_full=run_dr_on_full_dataset, run_dsa=run_data_scarcity_analysis)

    if run_model:
        print('\n*** MODELLING PHASE ***')
        run_models()

    if run_error_analysis:
        print('\n*** ERROR ANALYSIS PHASE ***')
        ea()


if __name__ == '__main__':
    print('> Starting main procedure')
    main(preprocessing_only=True,
         run_error_analysis=True,
         run_model=True,
         run_wrap_methods=True,
         run_dr_on_full_dataset=False,
         run_data_scarcity_analysis=False)
