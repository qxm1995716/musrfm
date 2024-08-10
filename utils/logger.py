import logging
from pathlib import Path
import datetime


class logger:
    def __init__(self, model_name):
        self.model_name = model_name
        self.Logger = logging.getLogger(model_name)
        self.Logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        'create log directory'
        self.experiment_dict = Path('./log/')
        self.experiment_dict.mkdir(exist_ok=True)

        'get current time'
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

        'build current directory'
        self.experiment_dict = self.experiment_dict.joinpath(model_name)
        self.experiment_dict.mkdir(exist_ok=True)
        self.experiment_dict = self.experiment_dict.joinpath(timestr)
        self.experiment_dict.mkdir(exist_ok=True)

        'build checkpoint directory'
        self.checkpoint_dict = self.experiment_dict.joinpath('checkpoints/')
        self.checkpoint_dict.mkdir(exist_ok=True)

        self.tif_dict = self.experiment_dict.joinpath('tif/')
        self.tif_dict.mkdir(exist_ok=True)

        'build log directory'
        self.log_dir = self.experiment_dict.joinpath('log/')
        self.log_dir.mkdir(exist_ok=True)

        '''
        'build dataset saved directory'
        self.data_dir = self.experiment_dict.joinpath('data/')
        self.data_dir.mkdir(exist_ok=True)
        
        'build onnx dir'
        self.onnx_dir = self.experiment_dict.joinpath('onnx/')
        self.onnx_dir.mkdir(exist_ok=True)
        '''

        'get file handler'
        self.file_handler = logging.FileHandler('%s/%s.txt' % (self.log_dir, model_name))
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(formatter)
        self.Logger.addHandler(self.file_handler)

        self.log_string('...Parameters...')

    def log_string(self, str):
        self.Logger.info(str)
        print(str)

