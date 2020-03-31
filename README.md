# HiLo-Network for 3D Baggage Segmentation
!!! PAPER CAN BE FOUND IN THIS BRANCH, TOO!!! <br />
Datasets are available on the lab 20 machine (fastdata). The correct paths are already used in the scripts if this repository is downloaded to your home directory. You may need to install pykdtree (https://github.com/storpipfugl/pykdtree), in case you want to check anything concerning Occupancy Networks. For the HiLo-Network cnn-approach this is not needed. 


This repo comes with 4 pretrained models, other models are available on request (due to memory consumption).

## Training
Training HiLo-Network using Occupancy Network
```python 
train_pipeline.py 
        -sl 16    # Length of patch sides
        -sld 8    # Length of patch sides divided by two
        -df 2     # Downsampling factor 
        -np 10    # Numper of points drawn for traning Occupancy Network
        -lr 3     # Learning Rate
        -ct fifo  # Queue Type
        -cr bce   # Loss function
        -l False  # Load from checkpoint
        -n occ    # Name for loading and saving checkpoints
```
Training HiLo-Network using CNN decoder
```python
train_pipeline_unet.py 
        -sl 16    # Length of patch sides
        -sld 8    # Length of patch sides divided by two\
        -df 8     # Downsampling factor 
        -lr 3     # Learning Rate
        -ct fifo  # Queue Type
        -cr focal # Loss function
        -l True   # Load from checkpoint
        -n        # Name for loading and saving checkpoints
 ```       
## Testing
Testing or drawing HiLo-Network using Occupancy Network decoder
```python
test_pipeline.py 
        -sl 16    # Length of patch sides
        -sld 8    # Length of patch sides divided by two
        -df 2     # Downsampling factor 
        -np 10    # Numper of points drawn for traning Occupancy Network
        -lr 3     # Learning Rate
        -cr bce   # Loss function
        -a test   # Wether to test or draw         
        -n occ    # Name for loading and saving checkpoints
```
Testing or drawing HiLo-Network using CNN (U-Net)
```python
test_pipeline_unet.py 
        -sl 16    # Length of patch sides
        -sld 8    # Length of patch sides divided by two\
        -df 2     # Downsampling factor 
        -lr 3     # Learning Rate
        -cr focal # Loss function
        -a test   # Wether to test or draw     
        -n        # Name for loading and saving checkpoints
```
## Generating
Generating of the various (decompressed) datasets
```python
data_interface.py 
        -a generate # (or check) action
        -r high     # (or low)   resolution
```
## Results
![](https://i.imgur.com/WTbDI4A.jpg)
