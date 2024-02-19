import time
from pathlib import Path
import os
from helper import project_path
import numpy as np
from dataclasses import dataclass
import json
from .plotter import Plotter


@dataclass
class ExperimentData:
    meta_data: dict = None
    data: dict = None

    def save_data(self, file_name: str = None):
        file_str = self.meta_data['type'] if file_name is None else file_name

        # Get the current date and time
        timestamp = time.strftime("%d_%m_%Y")

        # Get the current working directory
        path = f'{project_path}/results/data/{timestamp}'
        Path(path).mkdir(parents=True, exist_ok=True)

        file_number = 1
        while True:
            file_name = f"{file_str}_{file_number}.json"
            file_path = os.path.join(path, file_name)
            if not os.path.exists(file_path):
                break
            file_number += 1

        self._convert_numpy_to_list()
        with open(file_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
        print(f'saved {file_path}')

    def _convert_numpy_to_list(self):
        converted_data = {}
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                converted_data[key] = value.tolist()
            else:
                converted_data[key] = value
        self.data = converted_data

    @staticmethod
    def load_data(date,experiment_name,number):

        file_path = f'{project_path}/results/data/{date}/{experiment_name}_{number}.json'

        with open(file_path, 'rb') as file:
            exp_data = json.load(file)
        print(f'loaded file: {file_path}')

        return exp_data


