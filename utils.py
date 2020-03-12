import pandas as pd
from sklearn.model_selection import train_test_split


COL_NAMES = [
    "review_profilename",
    "beer_beerid",
    "review_overall",
    "review_aroma",
    "review_appearance",
    "review_palate",
    "review_taste"
]


def load_beers(path):
    """
    Loads the database.

    Parameters:
    path (string) -- the file's path.

    Returns:
    (pandas.DataFrame) -- the source data set.
    """
    return pd.read_csv(path)[COL_NAMES].rename(columns={
        COL_NAMES[0]: "user_id",
        COL_NAMES[1]: "beer_id"
    })


def split_data(source, usr_th=350, itm_th=350):
    """
    Splits the data into the training, validation and testing sets
    in a 80:10:10 ratio, respectively.

    Parameters:
    source (matrix-like) -- the source data set.
    usr_th (integer) -- the minimum number of transactions of a valid user.
    itm_th (integer) -- the minimum number of transactions of a valid item.

    Returns (3-tuple):
    (matrix-like) -- the training set.
    (matrix-like) -- the validation set.
    (matrix-like) -- the testing set.
    """
    usr_mask = source.groupby("user_id")["user_id"].transform(
        "count").ge(usr_th)
    source = source[usr_mask]
    itm_mask = source.groupby("beer_id")["beer_id"].transform(
        "count").ge(itm_th)
    source = source[itm_mask]
    train, rest = train_test_split(source, test_size=0.2)
    val, test = train_test_split(rest, test_size=0.5)
    return train, val, test
