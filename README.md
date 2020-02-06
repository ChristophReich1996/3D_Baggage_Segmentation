# HiLo-Network for 3D Baggage Segmentation
## Training
```python train_pipeline.py \
        -sl 16    # Length of patch sides\
        -sld 8    # Length of patch sides divided by two\ 
        -df 2     # Downsampling factor \
        -np 10    # Numper of points drawn for traning Occupancy Network\
        -lr 3     # Learning Rate\
        -ct fifo  # Queue Type\
        -cr bce   # Loss function\
        -l False  # Load from checkpoint\
        -n occ    # Name for loading and saving checkpoints
```

## Testing

## Results
