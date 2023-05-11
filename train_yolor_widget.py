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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from train_yolor.train_yolor_process import Param

#PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class Widget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = Param()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Model name
        self.combo_model_name = pyqtutils.append_combo(self.grid_layout, "Model name")
        self.combo_model_name.addItem("yolor_p6")
        self.combo_model_name.addItem("yolor_w6")
        self.combo_model_name.setCurrentText(self.parameters.cfg["model_name"])

        # Epochs
        self.spin_epochs = pyqtutils.append_spin(self.grid_layout, "Epochs", self.parameters.cfg["epochs"])

        # Batch size
        self.spin_batch = pyqtutils.append_spin(self.grid_layout, "Batch size", self.parameters.cfg["batch_size"])

        # Input size
        self.spin_train_img_size = pyqtutils.append_spin(self.grid_layout, "Train image size",
                                                         self.parameters.cfg["train_imgsz"])
        self.spin_test_img_size = pyqtutils.append_spin(self.grid_layout, "Test image size",
                                                        self.parameters.cfg["test_imgsz"])

        # Evaluation period
        self.spin_eval_period = pyqtutils.append_spin(self.grid_layout, "Eval period", self.parameters.cfg["eval_period"])

        # Ratio train test
        self.spin_train_test = pyqtutils.append_spin(self.grid_layout, "Train test percentage", self.parameters.cfg["dataset_split_ratio"])

        # Hyper-parameters
        custom_hyp = self.parameters.cfg["config_hyper_param_file"] != ""
        self.check_hyp = QCheckBox("Custom hyper-parameters")
        self.check_hyp.setChecked(custom_hyp)
        self.grid_layout.addWidget(self.check_hyp, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_hyp.stateChanged.connect(self.on_custom_hyp_changed)

        self.label_hyp = QLabel("Hyper-parameters file")
        self.browse_hyp_file = pyqtutils.BrowseFileWidget(path=self.parameters.cfg["config_hyper_param_file"],
                                                          tooltip="Select file",
                                                          mode=QFileDialog.ExistingFile)

        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_hyp, row, 0)
        self.grid_layout.addWidget(self.browse_hyp_file, row, 1)

        self.label_hyp.setVisible(custom_hyp)
        self.browse_hyp_file.setVisible(custom_hyp)

        # Custom Model
        custom_model = self.parameters.cfg["config_file"] != ""
        self.check_model = QCheckBox("Custom model")
        self.check_model.setChecked(custom_model)
        self.grid_layout.addWidget(self.check_model, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_model.stateChanged.connect(self.on_use_custom_model_changed)

        self.label_model = QLabel("Model config file")
        self.browse_cfg_file = pyqtutils.BrowseFileWidget(path=self.parameters.cfg["config_file"],
                                                          tooltip="Select file",
                                                          mode=QFileDialog.ExistingFile)

        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_model, row, 0)
        self.grid_layout.addWidget(self.browse_cfg_file, row, 1)

        self.label_model.setVisible(custom_model)
        self.browse_cfg_file.setVisible(custom_model)

        # Pretrain
        pretrain = self.parameters.cfg["model_weight_file"] != ""
        self.check_pretrain = QCheckBox("Pretrain")
        self.check_pretrain.setChecked(pretrain)
        self.grid_layout.addWidget(self.check_pretrain, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_pretrain.stateChanged.connect(self.on_pretrain_changed)

        self.label_pretrain = QLabel("Pretrain weight file")
        self.browse_pretrain_file = pyqtutils.BrowseFileWidget(path=self.parameters.cfg["model_weight_file"],
                                                          tooltip="Select file",
                                                          mode=QFileDialog.ExistingFile)

        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_pretrain, row, 0)
        self.grid_layout.addWidget(self.browse_pretrain_file, row, 1)

        self.label_pretrain.setVisible(pretrain)
        self.browse_pretrain_file.setVisible(pretrain)

        # Output folder
        self.browse_out_folder = pyqtutils.append_browse_file(self.grid_layout, label="Output folder",
                                                              path=self.parameters.cfg["output_folder"],
                                                              tooltip="Select folder",
                                                              mode=QFileDialog.Directory)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_custom_hyp_changed(self, int):
        self.label_hyp.setVisible(self.check_hyp.isChecked())
        self.browse_hyp_file.setVisible(self.check_hyp.isChecked())

    def on_use_custom_model_changed(self, int):
        self.label_model.setVisible(self.check_model.isChecked())
        self.browse_cfg_file.setVisible(self.check_model.isChecked())

    def on_pretrain_changed(self, int):
        self.label_pretrain.setVisible(self.check_pretrain.isChecked())
        self.browse_pretrain_file.setVisible(self.check_pretrain.isChecked())

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.cfg["model_name"] = self.combo_model_name.currentText()
        self.parameters.cfg["epochs"] = self.spin_epochs.value()
        self.parameters.cfg["batch_size"] = self.spin_batch.value()
        self.parameters.cfg["train_imgsz"] = self.spin_train_img_size.value()
        self.parameters.cfg["test_imgsz"] = self.spin_test_img_size.value()
        self.parameters.cfg["eval_period"] = self.spin_eval_period.value()
        self.parameters.cfg["dataset_split_ratio"] = self.spin_train_test.value()

        if self.check_hyp.isChecked():
            self.parameters.cfg["config_hyper_param_file"] = self.browse_hyp_file.path
        else:
            self.parameters.cfg["config_hyper_param_file"] = ""
        if self.check_model.isChecked():
            self.parameters.cfg["config_file"] = self.browse_cfg_file.path
        else:
            self.parameters.cfg["config_file"] = ""
        if self.check_pretrain.isChecked():
            self.parameters.cfg["model_weight_file"] = self.browse_pretrain_file.path
        else:
            self.parameters.cfg["model_weight_file"] = ""

        self.parameters.cfg["output_folder"] = self.browse_out_folder.path

        # Send signal to launch the process
        self.emit_apply(self.parameters)



# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class WidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "train_yolor"

    def create(self, param):
        # Create widget object
        return Widget(param, None)
