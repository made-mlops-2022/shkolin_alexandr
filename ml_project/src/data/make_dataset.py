import sys
import logging
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def read_raw_data(path: str) -> pd.DataFrame:
    """Read data from csv file"""

    logger.info(f"Loading dataset from {path}...")

    df = pd.read_csv(path)

    logger.info("Data loaded")
    logger.info(f"Data shape: {df.shape}")

    return df


def prepare_data(df: pd.DataFrame, params) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Prepare data"""

    logger.info('Preparing dataset...')

    categorical_columns = ['hotel', 'arrival_date_month', 'meal', 'country',
                           'market_segment', 'distribution_channel', 'reserved_room_type',
                           'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status_date']
    df = df.drop(categorical_columns, axis=1)

    if params["target"] in df.columns:
        logger.info('Extract target...')

        target = df[params["target"]]
        df = df.drop([params["target"]], axis=1)

        return df, target

    return df, None


def split_data(df: pd.DataFrame,
               target: pd.Series,
               params) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data to train and test"""

    x_train, x_test, y_train, y_test = train_test_split(df,
                                                        target,
                                                        test_size=params["test_size"],
                                                        random_state=params["random_state"])
    logger.info('Splitting data finished')

    return x_train, x_test, y_train, y_test
