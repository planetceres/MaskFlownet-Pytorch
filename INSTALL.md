# Install


Requirements:

* PyTorch 1.5.0
* torchvision 0.6.0
* CUDA 10.1
* OpenCV 4+
* gcc/g++ 5


NOTE: gcc/g++ version required for builidng `correlation_package` 



Conda:

```
conda create --name maskflownet-37 python=3.7 -y
conda activate maskflownet-37
conda install pytorch=1.5.0 torchvision=0.6.0 cudatoolkit=10.1 -c pytorch
conda install matplotlib tensorboard scipy opencv
conda install -c conda-forge opencv=4.1.0
pip3 install pyrealsense2 imutils pyyaml
```



To Run:

```
python demo_rs.py MaskFlownet.yaml -c 5adNov03-0005_1000000.pth --dataset_cfg sintel.yaml
python demo_rs.py MaskFlownet.yaml -c 8caNov12-1532_300000.pth --dataset_cfg kitti.yaml
python demo_rs.py MaskFlownet_S.yaml -c 771Sep25-0735_500000.pth --dataset_cfg chairs.yaml
python demo_rs.py MaskFlownet_S.yaml -c dbbSep30-1206_1000000.pth --dataset_cfg sintel.yaml

```
