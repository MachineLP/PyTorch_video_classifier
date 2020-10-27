
ref： https://github.com/jfzhang95/pytorch-video-recognition 

## Installation
The code was tested with Anaconda and Python 3.5. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/MachineLP/Pytorch_video_classifier.git
    cd Pytorch_video_classifier
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    conda install opencv
    pip install tqdm scikit-learn tensorboardX
    ```

2. Download pretrained model from [BaiduYun](https://pan.baidu.com/s/1saNqGBkzZHwZpG-A5RDLVw) or 
[GoogleDrive](https://drive.google.com/file/d/19NWziHWh1LgCcHU34geoKwYezAogv9fX/view?usp=sharing).
   Currently only support pretrained model for C3D.

3. Configure your dataset and pretrained model path in
[mypath.py](./mypath.py).

4. You can choose different models and datasets in
[train.py](./train.py).

    To train the model, please do:
    ```Shell
    python train.py
    ```
5. inference
    To infer the model, please do:
    ```Shell
    python inference.py
    ```
