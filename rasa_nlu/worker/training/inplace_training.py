import json
import tempfile
import functools

from future.utils import PY3

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import InvalidProjectError
from rasa_nlu.train import do_train_in_worker


class InplaceTraining(object):
    @classmethod
    def train(cls, config, data):
        data_file = cls._write_training_data_to_file(data)
        train_config = cls.update_training_config(config, data_file)

        cls.check_project(train_config)
        cls.do_train(train_config)

    @staticmethod
    def _write_training_data_to_file(data):
        if isinstance(data, dict):
            # convert dict to string
            data = json.dumps(data)

        if PY3:
            f = tempfile.NamedTemporaryFile("w+", suffix="_training_data",
                                            delete=False, encoding="utf-8")
            f.write(data)
        else:
            f = tempfile.NamedTemporaryFile("w+", suffix="_training_data",
                                            delete=False)
            f.write(data.encode("utf-8"))
        f.close()

        return f.name

    @staticmethod
    def update_training_config(config, data_file):
        config["data"] = data_file
        return RasaNLUConfig(cmdline_args=config)

    @staticmethod
    def check_project(config):
        project = config.get("project")
        if not project:
            raise InvalidProjectError("Missing project name to train")

    @staticmethod
    def do_train(train_config):
        do_train_in_worker(train_config)


inplace_train = InplaceTraining.train
