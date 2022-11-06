from dimensionality_reduction.dimensionality_reduction_on_feature_selection.PCA import main as pca
from dimensionality_reduction.dimensionality_reduction_on_feature_selection.PCA_visualization import main as pca_viz
from dimensionality_reduction.dimensionality_reduction_on_feature_selection.data_scarcity_analysis import main as dsa
from dimensionality_reduction.dimensionality_reduction_on_feature_selection.t_sne import main as tsne
from dimensionality_reduction.dimensionality_reduction_on_feature_selection.t_sne_visualization import main as tsne_viz
from dimensionality_reduction.dimensionality_reduction_on_full_dataset.PCA import main as full_pca
from dimensionality_reduction.dimensionality_reduction_on_full_dataset.PCA_visualization import main as full_pca_viz
from dimensionality_reduction.dimensionality_reduction_on_full_dataset.data_scarcity_analysis import main as full_dsa
from dimensionality_reduction.dimensionality_reduction_on_full_dataset.t_sne import main as full_tsne
from dimensionality_reduction.dimensionality_reduction_on_full_dataset.t_sne_visualization import main as full_tsne_viz


def main(run_dsa=False, run_full=False):
    if run_dsa:
        dsa()
    print('\n> Running PCA')
    pca()
    print('\n> Running t-SNE')
    tsne()
    pca_viz()
    tsne_viz()
    if run_full:
        if run_dsa:
            full_dsa()
        print('\n> Running PCA on full dataset')
        full_pca()
        print('\n> Running t-SNE on full dataset')
        full_tsne()
        full_pca_viz()
        full_tsne_viz()


if __name__ == '__main__':
    main(run_full=False)
