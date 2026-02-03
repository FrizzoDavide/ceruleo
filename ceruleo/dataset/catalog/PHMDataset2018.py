import ipdb
import gzip
import io
import logging
import os
import pickle
import shutil
import tarfile
import zipfile
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import gdown
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ceruleo import CACHE_PATH, DATA_PATH
from ceruleo.dataset.builder.builder import DatasetBuilder
from ceruleo.dataset.builder.cycles_splitter import FailureDataCycleSplitter
from ceruleo.dataset.builder.output import DatasetFormat, LocalStorageOutputMode
from ceruleo.dataset.builder.rul_column import NumberOfRowsRULColumn
from ceruleo.dataset.ts_dataset import AbstractPDMDataset, PDMDataset

logger = logging.getLogger(__name__)

COMPRESSED_FILE = "phm_data_challenge_2018.tar.gz"
ZIP_COMPRESSED_FILE = "phm_data_challenge_2018.zip"
FOLDER = "phm_data_challenge_2018"


#NOTE: Old link
# URL = "https://drive.google.com/uc?id=15Jx9Scq9FqpIGn8jbAQB_lcHSXvIoPzb"

#NOTE: New link
URL = "https://drive.google.com/uc?id=19e5OjnLY8gKXChzVBqPgRtJzq4TxVZy7"

OUTPUT = ZIP_COMPRESSED_FILE


def download(url: str, path: Path):
    logger.info("Downloading dataset...")
    gdown.download(url, str(path / OUTPUT), quiet=False)

def get_key_from_filename(filename: str) -> str:
    return "_".join(filename.split("_")[0:2])

PHM_TOOLS = [
    "01M01",
    "01M02",
    "02M01",
    "02M02",
    "03M01",
    "03M02",
    "04M01",
    "04M02",
    "05M01",
    "05M02",
    "06M01",
    "06M02",
    "07M01",
    "07M02",
    "08M01",
    "08M02",
    "09M01",
    "09M02",
    "10M01",
    "10M02",
]

PHM_TEST_TOOLS = [
    "01M02",
    "02M02",
    "03M01",
    "04M01",
    "06M01",
]

class FailureType(Enum):
    """Failure types availables for the dataset.

    Possible values are:
    ```
    FailureType.FlowCoolPressureDroppedBelowLimit
    FailureType.FlowcoolPressureTooHighCheckFlowcoolPump
    FailureType.FlowcoolLeak
    ```
    """

    FlowCoolPressureDroppedBelowLimit = "FlowCool Pressure Dropped Below Limit"
    FlowcoolPressureTooHighCheckFlowcoolPump = (
        'Flowcool Pressure Too High Check Flowcool Pump'
    )
    FlowcoolLeak = "Flowcool leak"
    FlowcoolPressureTooHighCheckFlowcoolPumpNoWaferID = 'Flowcool Pressure Too High Check Flowcool Pump [NoWaferID]'


    @staticmethod
    def that_starth_with(s: str):
        for f in FailureType:
            if s.startswith(f.value):
                return f
        return None

class PHMDataset2018(PDMDataset):
    """PHM 2018 Dataset

    The 2018 PHM dataset is a public dataset released by Seagate which contains the execution of 20 different
    ion milling machines. They distinguish three different failure causes and provide 22 features,
    including user-defined variables and sensors.

    Three faults are present in the dataset

    * Fault mode 1 occurs when flow-cool pressure drops.
    * Fault mode 2 occurs when flow-cool pressure becomes too high.
    * Fault mode 3 represents flow-cool leakage.

    [Dataset reference](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/)

    Example:

    ```py
    dataset = PHMDataset2018(
        failure_types=FailureType.FlowCoolPressureDroppedBelowLimit,
        tools=['01_M02']
    )
    ```

    Parameters:
        path: Path where the dataset is located, by default DATA_PATH
        url: url containing the data in a zip file, by default URL
        failure_types: failure type to consider, by default None
        tools: phm tools (i.e ion milling machines) to consider, by default None
        train: boolean flag to decide weather to load the trainig or test data, by default True
    """

    failure_types: Optional[List[FailureType]]
    tools: Optional[List[str]]


    def __init__(
        self,
        path: Path = DATA_PATH,
        url: str = URL,
        failure_types: Optional[Union[FailureType, List[FailureType]]] = None,
        tools: Optional[Union[str, List[str]]] = None,
        train: bool = True
    ):
        self.url = url
        self.failure_types = failure_types
        self.tools = tools
        self.train = train
        self.dataset_path = path / "phm_data_challenge_2018"
        self.procesed_path = self.dataset_path / "processed" / "train_cycles" if self.train else self.dataset_path / "processed" / "test_cycles"
        self.cycles_table_filename = self.procesed_path / "cycles.csv"

        super().__init__(path / "phm_data_challenge_2018", "RUL")
        self.dataset_path = path

        if self.failure_types is not None:
            if not isinstance(self.failure_types, list):
                self.failure_types = [failure_types]

            self.cycles_metadata = self.cycles_metadata[
                self.cycles_metadata["Fault name"].isin(
                    [f.value for f in self.failure_types]
                )
            ]

        if self.tools is not None:
            if not isinstance(self.tools, list):
                self.tools = [tools]

            if self.train:
                assert set(self.tools).issubset(PHM_TOOLS), f"Some of the tools defined in {self.tools} are not available. Available tools are {PHM_TOOLS}"
            else:
                assert set(self.tools).issubset(PHM_TEST_TOOLS), f"Some of the tools defined in {self.tools} are not available for the test set. Available test tools are {PHM_TEST_TOOLS}"

            self.cycles_metadata = self.cycles_metadata[self.cycles_metadata["Tool"].isin(self.tools)]

    def _extract_lifes(
        self,
        files: list,
        faults_files: list
    ):
        """
        Extract lifes from the raw data

        Args:
            files: list of csv files containing the raw sensor data
            faults_files: list of csv files containingt the fault times for each tool
        """

        fault_files_map = {get_key_from_filename(f.name): f for f in faults_files}
        data_fault_pairs = [
            (file, fault_files_map[get_key_from_filename(file.name)]) for file in files
        ]

        (
            DatasetBuilder()
            .set_splitting_method(
                FailureDataCycleSplitter(
                    data_time_column="time", fault_time_column="time"
                )
            )
            .set_rul_column_method(NumberOfRowsRULColumn())
            .set_output_mode(
                LocalStorageOutputMode(
                    output_path = self.dataset_path,
                    output_format=DatasetFormat.PARQUET,
                    train = self.train
                ).set_metadata_columns(
                    {"Tool": "Tool_data", "Fault name": "fault_name"}
                )
            )
            .set_index_column("time")
            .prepare_from_data_fault_pairs_files(
                data_fault_pairs,
            )
        )

    def _prepare_dataset(self):

        if self.cycles_table_filename.is_file():
            return

        if not (self.dataset_path / "raw" / "train").is_dir():
            self.prepare_raw_dataset()

        if self.train:

            files = list(Path(self.dataset_path / "raw" / "train").resolve().glob("*.csv"))
            faults_files = list(
                Path(self.dataset_path / "raw" / "train" / "train_faults")
                .resolve()
                .glob("*.csv")
            )

        else:

            files = list(Path(self.dataset_path / "raw" / "test").resolve().glob("*.csv"))
            faults_files = list(
                Path(self.dataset_path / "raw" / "test" / "test_faults")
                .resolve()
                .glob("*.csv")
            )

        self._extract_lifes(files=files,faults_files=faults_files)


    def prepare_raw_dataset(self):
        """Download and unzip the raw files

        Args:
            path (Path): Path where to store the raw dataset
        """

        def track_progress(members):
            for member in tqdm(members, total=70):
                yield member

        path = self.dataset_path / "raw"
        path.mkdir(parents=True, exist_ok=True)
        archive_path = path / OUTPUT
        if not (archive_path).resolve().is_file():
            download(self.url, path)

        is_zip = zipfile.is_zipfile(archive_path)

        if is_zip:

            logger.info("Decompressing  dataset with zip...")
            phm_dirname = "phm_data_challenge_2018_complete"

            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                for member in zip_ref.namelist():
                    if os.path.isabs(member) or ".." in member:
                        raise Exception("Attempted Path Traversal in Zip File")

                zip_ref.extractall(path)

        else:

            logger.info("Decompressing  dataset with tar...")
            phm_dirname = "phm_data_challenge_2018"

            with tarfile.open(path / OUTPUT, "r") as tarball:

                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")

                    tar.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(tarball, path=path, members=track_progress(tarball))

        shutil.move(str(path / phm_dirname / "train"), str(path / "train"))
        shutil.move(str(path / phm_dirname / "test"), str(path / "test"))
        shutil.move(str(path / phm_dirname / "test_after"), str(path / "test_after"))
        shutil.move(str(path / phm_dirname / "test_concat"), str(path / "test_concat"))
        shutil.rmtree(str(path / phm_dirname))
        (path / OUTPUT).unlink()
