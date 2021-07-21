# HRGAT
The code for hyper-node relational graph attention network

## requirement
torch == 1.7.0
dgl == 0.4.2
sentence-bert

### Train Model
To start a simple training process:

```shell script
python run.py --data FB15k-237 --text --img --attr
```

  - `--model` denotes the link prediction score score function  
  - `--gpu` for specifying the GPU to use
  - `--epoch` for number of epochs
  - `--batch` for batch size
  - `--text` for text information
  - `--img` for batch information
  - `--attr` for batch information
  - Rest of the arguments can be listed using `python run.py -h`
