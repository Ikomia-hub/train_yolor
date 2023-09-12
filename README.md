<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_yolor/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">train_yolor</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_yolor">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_yolor">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_yolor/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_yolor.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Train YoloR object detection models.

![YOLOR illustration](https://github.com/WongKinYiu/yolor/blob/main/inference/output/horses.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "C:/Users/allan/OneDrive/Desktop/ik-desktop/Images/datasets/wgisd/coco_annotations/test_polygons_instances.json",
    "image_folder": "C:/Users/allan/OneDrive/Desktop/ik-desktop/Images/datasets/wgisd/data",
    "task": "detection",
}) 

train = wf.add_task(name="train_yolor", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters


- **model_name** (str) - default 'yolor_p6': Name of the pre-trained model. Other model: "yolor_w6"
- **epochs** (int) - default '50': Number of complete passes through the training dataset.
- **batch_size** (int) - default '8': Number of samples processed before the model is updated.
- **train_imgsz** (int) - default '512': Size of the training image.
- **test_imgsz** (int) - default '512': Size of the eval image.
- **dataset_split_ratio** (float) – default '90': Divide the dataset into train and evaluation sets ]0, 100[.
- **eval_period** (int) - default '5': Interval between evaluations.  
- **output_folder** (str, *optional*): path to where the model will be saved. 



**Parameters** should be in **strings format**  when added to the dictionary.


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "C:/Users/allan/OneDrive/Desktop/ik-desktop/Images/datasets/wgisd/coco_annotations/test_polygons_instances.json",
    "image_folder": "C:/Users/allan/OneDrive/Desktop/ik-desktop/Images/datasets/wgisd/data",
    "task": "detection",
}) 

train = wf.add_task(name="train_yolor", auto_connect=True)
train.set_parameters({
    "model_name": "yolor_p6",
    "epochs": "5",
    "batch_size": "4",
    "input_width": "512",
    "input_height": "512",
    "dataset_split_ratio": "90"
}) 

# Launch your training on your data
wf.run()
```
