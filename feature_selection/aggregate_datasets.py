# Auxiliary library to combine the results from the wrapper methods
# based on feature commonality and performance with the tuned XGB model
# with embedded methods based on feature importance.

# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Driver:
if __name__ == '__main__':
    # best performance from the embedded method on the full dataset selects the following features:
    """"
    Recency, 0.46658403  RFM
    1_Area under the curve, 0.16938186 TS
    1_Spectral distance, 0.14503708 TS
    1_Autocorrelation, 0.06448283 TS
    1_ECDF Percentile Count_0, 0.04268373 TS
    1_ECDF Percentile Count_1, 0.042546716 TS
    1_Centroid, 0.025556285 TS
    94, 0.019752635 GR
    9, 0.009465399 GR 
    0_Min, 0.0029282405 TS
    101_country, 0.002909824 GR
    48, 0.0027074742 GR
    71_country, 0.0013282576 GR
    51, 0.0012885486 GR
    7, 0.0012776888 GR
    0_FFT mean coefficient_3, 0.00107738 TS
    2, 0.0009919758 GR
    """
    # load the dataset for feature selection:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_for_fs.csv'))

    # select the features named above:
    X = df[['Recency', '1_Area under the curve', '1_Spectral distance', '1_Autocorrelation',
            '1_ECDF Percentile Count_0', '1_ECDF Percentile Count_1', '1_Centroid', '94', '9', '0_Min',
            '101_country', '48', '71_country', '51', '7', '0_FFT mean coefficient_3', '2']]

    # rename some columns:
    X = X.rename(columns={'1_Area under the curve': 'AverageDaysBetweenPurchaseAUC',
                          '1_Spectral distance': 'AverageDaysBetweenPurchaseSpectralDistance',
                          '1_Autocorrelation': 'AverageDaysBetweenPurchaseAutocorrelation',
                          '1_ECDF Percentile Count_0': 'AverageDaysBetweenPurchaseECDFPercentileCount0',
                          '1_ECDF Percentile Count_1': 'AverageDaysBetweenPurchaseECDFPercentileCount1',
                          '1_Centroid': 'AverageDaysBetweenPurchaseCentroid',
                          '0_Min': 'MinTotalSpent',
                          '0_FFT mean coefficient_3': 'TotalSpentFFTMeanCoefficient3',
                          '101_country': 'DeepwalkCountry101',
                          '71_country': 'DeepwalkCountry71',
                          '48': 'DeepwalkCustomer48',
                          '51': 'DeepwalkCustomer51',
                          '94': 'DeepwalkCustomer94',
                          '9': 'DeepwalkCustomer9',
                          '7': 'DeepwalkCustomer7',
                          '2': 'DeepwalkCustomer2'})

    # save the dataset:
    X.to_csv(Path('..', 'data', 'online_sales_dataset_for_dr.csv'), index=False)
