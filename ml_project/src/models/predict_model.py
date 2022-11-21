import sys
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_model(model: LogisticRegression, x: pd.DataFrame) -> np.ndarray:
    """Predict target"""

    logger.info("Predicting target...")

    result = model.predict(x)

    logger.info("Target predicted.")

    return result
