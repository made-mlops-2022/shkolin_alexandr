import sys
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_model(x: pd.DataFrame, y: pd.Series, train_params) -> LogisticRegression:
    """Train model with the given params"""

    logger.info(f"Params: {train_params}. Loading model...")

    model = LogisticRegression(penalty=train_params["penalty"], solver=train_params["solver"],
                               max_iter=train_params["max_iter"])

    logger.info("Model loaded.")
    logger.info("Fitting model...")

    model.fit(x, y)

    logger.info('Model fitted.')

    return model
