# Libraries:
import pandas as pd


# Functions:
def cancelling_order_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes the cancelled orders in the dataset.
    @param df: dataframe with the cancelled orders.
    :return: dataframe with the cancelled orders imputed.
    """
    # create a list of cancelled invoices, removing the C from the invoice number:
    cancelled_invoices = df[df['Invoice'].str.startswith('C')]['Invoice'].str[1:].tolist()
    # filter out the unmatched cancelled invoices:
    invoices = df['Invoice'].tolist()
    unmatched_invoices = [i for i in cancelled_invoices if i not in invoices]
    # delete the unmatched cancelling invoices adding back the C:
    df.drop(df[df['Invoice'].isin([f'C{i}' for i in unmatched_invoices])].index, inplace=True)
    # filter out the matched cancelling invoices:
    matched_invoices = [i for i in cancelled_invoices if i not in unmatched_invoices]

    # look for partial cancellations:
    for i in matched_invoices:
        if df[df['Invoice'] == i]['Quantity'].sum() >= df[df['Invoice'] == f'C{i}']['Quantity'].sum() and \
                df[df['Invoice'] == i]['StockCode'].tolist() == df[df['Invoice'] == f'C{i}']['StockCode'].tolist():
            # if the quantity in the cancelled invoice is greater or equal to the quantity in the original invoice
            # and the stock code matches, then delete the original invoice:
            df.drop(df[df['Invoice'] == i].index, inplace=True)
        else:
            # if the quantity in the cancelled invoice is less than the quantity in the original invoice, then
            # subtract the quantity in the cancelled invoice from the quantity in the original invoice:
            df[df['Invoice'] == i]['Quantity'] = df[df['Invoice'] == i]['Quantity'] - \
                                                 df[df['Invoice'] == f'C{i}']['Quantity']
            # delete all rows with cancelling invoices:
            df.drop(df[df['Invoice'] == f'C{i}'].index, inplace=True)

        df = df[~df['Invoice'].str.startswith('C')]

    return df


def missing_descriptions_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes the missing descriptions in the dataset that can be recovered with the stock code, removes the rest.
    @param df: dataframe with the missing descriptions.
    :return: dataframe with the missing descriptions imputed.
    """
    # get the rows with missing descriptions but with a matching stock code with a non-missing description:
    missing_descriptions_with_matching_stock_code = df[df['Description'].isna()
                                                       & df['StockCode'].isin(df[df['Description']
                                                                              .notna()]['StockCode'])].index
    for i in missing_descriptions_with_matching_stock_code:
        df.loc[i, 'Description'] = df[df['StockCode'] == df.loc[i, 'StockCode']]['Description'].values[0]
    # Drop the remaining missing descriptions which cannot be recovered:
    df.dropna(subset=['Description'], inplace=True)

    return df
