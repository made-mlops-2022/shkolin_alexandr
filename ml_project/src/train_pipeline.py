import sys
import logging
import click

from models import train_model, predict_model, save_model
from entity import read_config
from data.make_dataset import read_raw_data, prepare_data, split_data

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    """Train pipeline"""

    logger.info("Start train pipeline...")

    training_pipeline_params = read_config(config_path)

    df = read_raw_data(training_pipeline_params["input_data_path"])
    df, target = prepare_data(df, training_pipeline_params["columns_params"])

    logger.info("Splitting data to train and test...")

    train_features, test_features, train_target, test_target = split_data(df,
                                                                          target,
                                                                          training_pipeline_params["splitting_params"]
                                                                          )

    logger.info(f"Train shape: {train_features.shape}")
    logger.info(f"Test shape: {test_features.shape}")

    model = train_model(train_features, train_target, training_pipeline_params["train_params"])
    predict = predict_model(model, test_features)

    """
    TO DO:
    Calculate and log metrics
    """

    logger.info("Model trained.")

    save_model(model, training_pipeline_params["save_model_path"])

    logger.info("Model saved.")

    return training_pipeline_params["save_model_path"]


@click.command(name='train_pipeline')
@click.argument('config_path', default='configs/train_lr_default.yaml')
def train_pipeline_command(config_path: str):
    """Make start by terminal"""

    train_pipeline(config_path)


if __name__ == '__main__':
    train_pipeline_command()
