## Spikeformer: A Novel Architecture for Training High-Performance Low-Latency Spiking Neural Network

For reproducibility, we provide our experimental configurations as follows.

```
NVIDIA GeForce RTX 3090

NVIDIA-SMI 495.44       Driver Version: 495.44       CUDA Version: 11.5

Linux version 5.4.0-65-generic (buildd@lcy01-amd64-018)

gcc version 9.3.0 (Ubuntu 9.3.0-17ubuntu1~20.04)
```

The trained models will be publicly available as well.

## Dependency

Note that although we used an identical seed during training, we found that the version of pytorch and even the version of CUDA driver will slightly influence the results. We present our training logs and tensorboard results in the ``logs`` folder.

To configure the python environment, we suggest to use anaconda.

```
conda create -n spikeformer python=3.8
conda activate spikeformer
pip install -r requirements.txt
```

## Reproducibility

### DVSCIFAR10

```bash
cd dvscifar10
python train.py
```

Both DVS-CIFAR10 and DVS-Gesture datasets should be obtained by referring to the document of spikingjelly.

The training period will takes up about 14G GPU memory. And each epoch will takes about 80s. Totally, it takes about 13.5 hours to achieve **81.4%** top-1 accuracy.

The final directory structure is shown as follows:

```
.
|-- DVSCIFAR
    |-- download
    |-- events_np
    |-- extract
    |-- frames_number_4_split_by_number
|-- logs
|-- neuron.py
|-- presets.py
|-- spikeformer.py
|-- train.py
|-- transforms.py
|-- utils.py
```



### DVS-Gesture

```bash
cd dvsgesture
python train.py
```

The training period will takes up about 21G GPU memory. And each epoch will takes about 47s. Totally, it takes about 2 hours to achieve **98.96%** top-1 accuracy.

The final directory structure is shown as follows:

```
.
|-- DVSGesture
    |-- download
    |-- events_np
    |-- extract
    |-- frames_number_16_split_by_number
|-- logs
|-- neuron.py
|-- presets.py
|-- spikeformer.py
|-- train.py
|-- transforms.py
|-- utils.py
```



### ImageNet

```bash
cd imagenet
torchrun --nproc_per_node=8 train.py --sync-bn
```

The training period will takes up about 22G GPU memory. And each epoch will takes about 1h2m7s. Totally, it takes about 5.6 days to achieve **78.31%** top-1 accuracy.

The final directory structure is shown as follows:

```
.
|-- ImageNet
    |-- train
        |-- n01440764
        |-- ...
        |-- n15075141
    |-- val
        |-- n01440764
        |-- ...
        |-- n15075141
|-- logs
|-- neuron.py
|-- presets.py
|-- spikeformer.py
|-- train.py
|-- transforms.py
|-- utils.py
```

Note that the validation set of ImageNet should be processed by ``valprep.sh``.

And we manually adjust the learning rate to {0.001, 0.0001, 0.00001} at epoch {94, 115, 118} respectively.

```bash
torchrun --nproc_per_node=8 train.py --sync-bn --lr 0.001 --sync-bn --resume /path/to/checkpoint --warmup -1
```

When resuming, the training script should be slightly modified. For the modification of the learning rate, line 350 and line 351 should be commented out. And in order to get consistent tensorboard training curve, the output directory should be modified (e.g. information about warmup and learning rate).