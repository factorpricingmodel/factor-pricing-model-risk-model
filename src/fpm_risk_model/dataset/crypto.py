import io
from os import listdir
from os.path import isfile, join
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd
import requests


def get_attr(data_all, field_name):
    """
    Get attribute for each value in the dictionary.

    :param data_all: Data in dict format of which its values
      are dataframes.
    """
    attr = {key: df[field_name] for key, df in data_all.items()}
    attr = pd.DataFrame(attr)
    attr = attr.ffill().groupby(attr.index.date).first()
    attr.index = pd.to_datetime(attr.index)
    attr = attr.groupby(attr.index.date).last()
    attr.index = pd.to_datetime(attr.index)
    return attr


def download_sample_data_estimation_universe():
    """
    Download sample data of estimation universe.

    :return: Dict of sample data with prices, returns, marketcap
      and volumes.
    """
    data = {}
    response = requests.get(
        "https://github.com/factorpricingmodel/factor-pricing-model-risk-model"
        "/raw/main/examples/notebook/crypto_estimation_universe.zip",
        stream=True,
    )
    with ZipFile(io.BytesIO(response.content)) as zip_ref:
        with TemporaryDirectory() as temp_dir:
            zip_ref.extractall(temp_dir)
            files = [
                (join(temp_dir, f), f.replace(".csv", ""))
                for f in listdir(temp_dir)
                if isfile(join(temp_dir, f))
            ]

            for file_path, name in files:
                df = pd.read_csv(file_path)
                df = df.set_index("date")
                df.index = pd.to_datetime(df.index)
                data[name] = df

    prices = get_attr(data, "price")
    returns = prices.pct_change()
    volumes = get_attr(data, "total_volume")
    marketcap = get_attr(data, "market_cap")

    return {
        "prices": prices,
        "returns": returns,
        "volumes": volumes,
        "marketcap": marketcap,
    }


def download_sample_data_model_universe():
    """
    Download sample data of model universe.

    :return: Dict of sample data with prices, returns, volumes
      (in crypto) and volumes (in USD).
    """
    data = {}
    response = requests.get(
        "https://github.com/factorpricingmodel/factor-pricing-model-risk-model"
        "/raw/main/examples/notebook/crypto_model_universe.zip",
        stream=True,
    )
    with ZipFile(io.BytesIO(response.content)) as zip_ref:
        with TemporaryDirectory() as temp_dir:
            zip_ref.extractall(temp_dir)
            price_data_dir = join(temp_dir, "Price-Data")
            files = [
                (join(price_data_dir, f), f.replace(".csv", ""))
                for f in listdir(price_data_dir)
                if isfile(join(price_data_dir, f))
            ]

            for file_path, name in files:
                df = pd.read_csv(file_path)
                df = df.set_index("Date")
                df.index = pd.to_datetime(df.index)
                data[name] = df

    prices = get_attr(data, "Adj Close")
    returns = prices.pct_change()
    volumes = get_attr(data, "Volume")
    volumes_usd = volumes * prices

    return {
        "prices": prices,
        "returns": returns,
        "volumes": volumes,
        "volumes_usd": volumes_usd,
    }
