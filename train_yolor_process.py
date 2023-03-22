# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import torch.cuda
from ikomia import core, dataprocess
from ikomia.core.task import TaskParam
from ikomia.dnn import datasetio, dnntrain

import copy
from train_yolor import train
from datetime import datetime
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from train_yolor.yolor_utils import change_cfg
import logging
# Your imports below

logger = logging.getLogger()


def init_logging(rank=-1):
    if rank in [-1, 0]:
        logger.handlers = []
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")

        info = logging.StreamHandler(sys.stdout)
        info.setLevel(logging.INFO)
        info.setFormatter(formatter)
        logger.addHandler(info)

        err = logging.StreamHandler(sys.stderr)
        err.setLevel(logging.ERROR)
        err.setFormatter(formatter)
        logger.addHandler(err)
    else:
        logging.basicConfig(format="%(message)s", level=logging.WARN)

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class Param(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        self.cfg["model_name"] = "yolor_p6"
        self.cfg["epochs"] = 50
        self.cfg["batch_size"] = 8
        self.cfg["train_img_size"] = 512
        self.cfg["test_img_size"] = 512
        self.cfg["dataset_split_ratio"] = 90
        self.cfg["custom_hyp_file"] = ""
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/runs/"
        self.cfg["custom_model"] = ""
        self.cfg["eval_period"] = 5
        self.cfg["pretrain"] = ""

    def set_values(self, param_map):
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["train_img_size"] = int(param_map["train_img_size"])
        self.cfg["test_img_size"] = int(param_map["test_img_size"])
        self.cfg["dataset_split_ratio"] = int(param_map["dataset_split_ratio"])
        self.cfg["custom_hyp_file"] = param_map["custom_hyp_file"]
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["custom_model"] = param_map["custom_model"]
        self.cfg["eval_period"] = int(param_map["eval_period"])
        self.cfg["pretrain"] = param_map["pretrain"]


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainProcess(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        # Create parameters class
        self.out_folder = None
        if param is None:
            self.set_param_object(Param())
        else:
            self.set_param_object(copy.deepcopy(param))
        self.stop_train = False

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.get_param_object()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()
        self.problem = False
        self.stop_train = False
        # Get parameters :
        param = self.get_param_object()

        input = self.get_input(0)

        # current datetime is used as folder name
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        if len(input.data) == 0:
            print("ERROR, there is no input dataset")
            self.problem = True
        else:
            classes = [v for k,v in input.data['metadata']['category_names'].items() ]

        # output dir
        self.out_folder = Path(param.cfg["output_folder"]) / str_datetime
        self.out_folder.mkdir(parents=True, exist_ok=True)

        # cfg
        if os.path.isfile(param.cfg["custom_model"]):
            self.cfg = Path(param.cfg["custom_model"])
        else:
            # get base cfg
            self.cfg = Path(os.path.dirname(os.path.realpath(__file__))+"/yolor/cfg/"+param.cfg["model_name"]+".cfg")
            model_name = param.cfg["model_name"]
            cfg_dst = self.out_folder / (model_name + ".cfg")
            nc = len(classes)
            change_cfg(self.cfg.__str__(),nc,cfg_dst.__str__())
            self.cfg = cfg_dst


        # hyp
        if os.path.isfile(param.cfg["custom_hyp_file"]):
            self.hyp = Path(param.cfg["custom_hyp_file"])
        else:
            self.hyp = Path(os.path.dirname(os.path.realpath(__file__))+"/yolor/data/hyp.scratch.640.yaml")

        # Tensorboard
        tb_logdir = os.path.join(core.config.main_cfg["tensorboard"]["log_uri"], str_datetime)
        tb_writer = SummaryWriter(tb_logdir)

        # image size
        self.img_size = (param.cfg["train_img_size"], param.cfg["test_img_size"])
        if not self.problem:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            train.train(data=input.data, save_dir=self.out_folder, epochs=param.cfg["epochs"],
                        eval_period=param.cfg["eval_period"], batch_size=param.cfg["batch_size"],
                        weights=param.cfg["pretrain"], cfg_file=self.cfg, hyp_file=self.hyp, device = device,
                        img_size=self.img_size, ratio_split_train_test=param.cfg["dataset_split_ratio"]/100,
                        tb_writer=tb_writer, stop=self.get_stop, emit_progress=self.emit_step_progress,
                        logger=logger, log_metrics=self.log_metrics)

        # Call end_task_run to finalize process
        self.end_task_run()

    def get_stop(self):
        return self.stop_train

    def stop(self):
        super().stop()
        self.stop_train = True


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class ProcessFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_yolor"
        self.info.short_description = "Train YoloR object detection models"
        self.info.description = "Train YoloR object detection models." \
                                "You Only Learn One Representation: Unified Network for Multiple Tasks"
        self.info.authors = "Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.0"
        self.info.icon_path = "icons/icon.png"
        self.info.article = "You Only Learn One Representation: Unified Network for Multiple Tasks"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "GPL-3.0 License"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2105.04206"
        # Code source repository
        self.info.repository = "https://github.com/WongKinYiu/yolor"
        # Keywords used for search
        self.info.keywords = "train, pytorch, object, detection"

    def create(self, param=None):
        # Create process object
        return TrainProcess(self.info.name, param)
