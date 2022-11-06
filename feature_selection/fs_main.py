from feature_selection.filtering.supervised.mutual_information import main as mi
from feature_selection.filtering.unsupervised.variance_threshold import main as variance_t
from feature_selection.fs_aggregate_datasets import main as fs_aggregator
from feature_selection.selected_features_merger import main as merger
from feature_selection.wrapper.backwards_selection import main as backward
from feature_selection.wrapper.correlation_fs_auto_analysis import main as corr
from feature_selection.wrapper.exhaustive_feature_selection import main as efs
from feature_selection.wrapper.forwards_selection import main as forward
from feature_selection.wrapper.recursive_feature_elimination import main as rfe


def main(run_wrappers=False, run_correlation=False):
    print('> creating variance threshold dataset')
    variance_t()
    print('\n> creating mutual information dataset')
    mi()
    if run_wrappers:
        print('\n> creating forward dataset')
        forward()
        print('\n> creating backward dataset')
        backward()
        print('\n> creating exhaustive dataset')
        efs()
        print('\n> creating recursive elimination dataset')
        rfe()
        merger()
    print('\n> aggregating feature selection dataset')
    fs_aggregator()
    if run_correlation:
        corr()


if __name__ == '__main__':
    main(run_wrappers=False, run_correlation=False)
