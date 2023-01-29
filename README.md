# Custom Training with YOLOv7 🔥

## Some Important links
- [Model Inference🤖](https://huggingface.co/spaces/owaiskha9654/Custom_Yolov7)
- [**🚀Training Yolov7 on Kaggle**](https://www.kaggle.com/code/owaiskhan9654/training-yolov7-on-kaggle-on-custom-dataset)
- [Weight and Biases 🐝](https://wandb.ai/owaiskhan9515/YOLOR)
- [HuggingFace 🤗 Model Repo](https://huggingface.co/owaiskha9654/Yolov7_Custom_Object_Detection)

## Contact Information


- **Name** - Owais Ahmad
- **Phone** - +91-9515884381
- **Email** - owaiskhan9654@gmail.com
- **Portfolio** - https://owaiskhan9654.github.io/

# Objective  

## To Showcase custom Object Detection on the Given Dataset to train and Infer the Model using newly launched YoloV7.

# Data Acquisition

The goal of this task is to train a model that
can localize and classify each instance of **Person** and **Car** as accurately as possible.

- [Link to the Downloadable Dataset](https://www.kaggle.com/datasets/owaiskhan9654/car-person-v2-roboflow)


```python
from IPython.display import Markdown, display

display(Markdown("../input/Car-Person-v2-Roboflow/README.roboflow.txt"))
```

# Custom Training with YOLOv7 🔥

In this Notebook, I have processed the images with RoboFlow because in COCO formatted dataset was having different dimensions of image and Also data set was not splitted into different Format.
To train a custom YOLOv7 model we need to recognize the objects in the dataset. To do so I have taken the following steps:

* Export the dataset to YOLOv7
* Train YOLOv7 to recognize the objects in our dataset
* Evaluate our YOLOv7 model's performance
* Run test inference to view performance of YOLOv7 model at work


# 📦 [YOLOv7](https://github.com/WongKinYiu/yolov7)
<div align=left><img src="https://raw.githubusercontent.com/WongKinYiu/yolov7/main/figure/performance.png" width=800>

    
**Image Credit** -  [WongKinYiu](https://github.com/WongKinYiu/yolov7)
</div>
# Step 1: Install Requirements


```python
!git clone https://github.com/WongKinYiu/yolov7 # Downloading YOLOv7 repository and installing requirements
%cd yolov7
!pip install -qr requirements.txt
!pip install -q roboflow

```

# **Downloading YOLOV7 starting checkpoint**


```python
!wget "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
```


```python
import os
import glob
import wandb
import torch
from roboflow import Roboflow
from kaggle_secrets import UserSecretsClient
from IPython.display import Image, clear_output, display  # to display images



print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```

<img src="https://camo.githubusercontent.com/dd842f7b0be57140e68b2ab9cb007992acd131c48284eaf6b1aca758bfea358b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67">

> I will be integrating W&B for visualizations and logging artifacts and comparisons of different models!
> 
> [YOLOv7-Car-Person-Custom](https://wandb.ai/owaiskhan9515/YOLOR)



```python
try:
    user_secrets = UserSecretsClient()
    wandb_api_key = user_secrets.get_secret("wandb_api")
    wandb.login(key=wandb_api_key)
    anonymous = None
except:
    wandb.login(anonymous='must')
    print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')
    
    
    
wandb.init(project="YOLOv7",name=f"7. YOLOv7-Car-Person-Custom-Run-7")
```

# Step 2: Assemble Our Dataset


![](https://uploads-ssl.webflow.com/5f6bc60e665f54545a1e52a5/615627e5824c9c6195abfda9_computer-vision-cycle.png)

In order to train our custom model, we need to assemble a dataset of representative images with bounding box annotations around the objects that we want to detect. And we need our dataset to be in YOLOv7 format.

In Roboflow, We can choose between two paths:

* Convert an existing Coco dataset to YOLOv7 format. In Roboflow it supports over [30 formats object detection formats](https://roboflow.com/formats) for conversion.
* Uploading only these raw images and annotate them in Roboflow with [Roboflow Annotate](https://docs.roboflow.com/annotate).



# Version v7 Jan 30, 2023 Looks like this.

![](https://raw.githubusercontent.com/Owaiskhan9654/Yolo-V7-Custom-Dataset-Train-on-Kaggle/main/Roboflow_train1.JPG)


### Since paid credits are required to train the model on RoboFlow I have used Kaggle Free resources to train it here 



### Note you can import any other data from other sources. Just remember to keep in the Yolov7 Pytorch form accept

![](https://raw.githubusercontent.com/Owaiskhan9654/Yolo-V7-Custom-Dataset-Train-on-Kaggle/main/Yolov7%20Pytorch%20format.JPG)

```python
user_secrets = UserSecretsClient()
roboflow_api_key = user_secrets.get_secret("roboflow_api")
```


```python
rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace("owais-ahmad").project("custom-yolov7-on-kaggle-on-custom-dataset-rakiq")
dataset = project.version(2).download("yolov7")
```

# Step 3: Training Custom pretrained YOLOv7 model

Here, I am able to pass a number of arguments:
- **img:** define input image size
- **batch:** determine batch size
- **epochs:** define the number of training epochs. (Note: often, 3000+ are common here nut since I am using free version of colab I will be only defining it to 20!)
- **data:** Our dataset locaiton is saved in the `./yolov7/Custom-Yolov7-on-Kaggle-on-Custom-Dataset-2` folder.
- **weights:** specifying a path to weights to start transfer learning from. Here I have choosen a generic COCO pretrained checkpoint.
- **cache:** caching images for faster training


```python
!python train.py --batch 16 --cfg cfg/training/yolov7.yaml --epochs 30 --data {dataset.location}/data.yaml --weights 'yolov7.pt' --device 0 

```

# Run Inference  With Trained Weights
Testing inference with a pretrained checkpoint on contents of `./Custom-Yolov7-on-Kaggle-on-Custom-Dataset-2/test/images` folder downloaded from Roboflow.


```python
!python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.75 --source ./Custom-Yolov7-on-Kaggle-on-Custom-Dataset-2/test/images
```

# Display inference on ALL test images


```python
for images in glob.glob('runs/detect/exp/*.jpg')[0:10]:
    display(Image(filename=images))
```


```python
model = torch.load('runs/train/exp/weights/best.pt')
```

# Conclusion and Next Steps

Now this trained custom YOLOv7 model can be used to recognize **Person** and **Cars** form any given Images.

To improve the model's performance, I might perform more interating on the datasets coverage,propper annotations and and Image quality. From orignal authors of **Yolov7** this guide has been given for [model performance improvement](https://github.com/WongKinYiu/yolov7).

To deploy our model to an application by [exporting your model to deployment destinations](https://github.com/WongKinYiu/yolov7/issues).

Once our model is in production, I will be willing  to continually iterate and improve on your dataset and model via [active learning](https://blog.roboflow.com/what-is-active-learning/).
