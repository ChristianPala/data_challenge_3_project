# Libraries
from preprocessing.pre_main import main as preprocessing
from feature_engineering.fe_main import main as feature_engineering
from feature_selection.fs_main import main as feature_selection
from dimensionality_reduction.dr_main import main as dimensionality_reduction
from error_analysis.error_analysis_with_shap_values import main as error_analysis
from modelling.model_main import main as run_models


def main(run_error_analysis=False, run_model=False):
    # running preprocessing (includes data loading and other)
    preprocessing(preprocessing_only=True)
    feature_engineering()
    feature_selection(run_wrappers=False, run_correlation=False)
    dimensionality_reduction(run_full=False, run_dsa=False)

    if run_model:
        run_models()

    if run_error_analysis:
        error_analysis()


if __name__ == '__main__':
    main(run_error_analysis=False, run_model=False)
