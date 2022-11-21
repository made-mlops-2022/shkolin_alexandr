import sys
import logging
import click
import pandas as pd

from models import predict_model, load_model
from data.make_dataset import read_raw_data, prepare_data

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline():
    """Predict pipeline"""

    logger.info('Start predict pipeline...')

    df = read_raw_data("data/raw/to_predict.csv")
    df, target = prepare_data(df, {"target": "is_canceled"})

    model = load_model("models/lr_default.pkl")

    logger.info('Making predict...')
    predict = predict_model(model, df)

    pd.Series(predict, index=df.index, name='Predict').to_csv("data/predictions/predict.csv")

    logger.info('Predict saved to csv.')


@click.command(name='predict_pipeline')
def predict_pipeline_command():
    """Make start by terminal"""

    predict_pipeline()


if __name__ == '__main__':
    predict_pipeline_command()
