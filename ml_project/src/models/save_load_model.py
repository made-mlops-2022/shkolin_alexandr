import sys
import logging

import pickle
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def save_model(model: LogisticRegression, path: str) -> str:
    """Save model as .pkl to path"""

    with open(path, "wb") as file:
        pickle.dump(model, file)

    logger.info("Model saved.")

    return path


def load_model(path: str) -> LogisticRegression:
    """Load .pkl file as model"""

    logger.info("Loading model...")

    with open(path, "rb") as model:
        return pickle.load(model)
