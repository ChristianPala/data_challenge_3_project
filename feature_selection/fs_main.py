from filtering.unsupervised.variance_threshold import main as variance_t
from wrapper.forwards_selection import main as forward
from wrapper.backwards_selection import main as backward
from wrapper.exhaustive_feature_selection import main as efs
from wrapper.recursive_feature_elimination import main as rfe
from filtering.supervised.mutual_information import main as mi
from wrapper.correlation_fs_auto_analysis import main as corr
from selected_features_merger import main as merger
from fs_aggregate_datasets import main as fs_aggregator


def main(run_wrappers=False, run_correlation=False):
    variance_t()
    mi()
    if run_wrappers:
        if run_correlation:
            corr()
        forward()
        backward()
        efs()
        rfe()
        merger()
    fs_aggregator()


if __name__ == '__main__':
    main(run_wrappers=False, run_correlation=False)
