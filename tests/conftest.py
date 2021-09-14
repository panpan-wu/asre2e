import csv
import os

import pytest


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


class TrainData:
    filepath = os.path.join(DATA_DIR, "train.csv")
    num_files = 0


@pytest.fixture(scope="session", autouse=True)
def gen_train_csv():
    src_path = os.path.join(DATA_DIR, "data_info.csv")
    dest_path = TrainData.filepath
    with open(src_path, "r") as f:
        with open(dest_path, "w") as t:
            reader = csv.reader(f)
            writer = csv.writer(t)
            for i, row in enumerate(reader):
                if i != 0:
                    row[1] = os.path.join(DATA_DIR, row[1])
                writer.writerow(row)
            TrainData.num_files = i
    yield
    os.remove(dest_path)
