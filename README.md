# CIFAR-ConvolutionalAutoEncoder-Chainer
Convolutional Autoencoder (CAE) implemented with Chainer

## Requirement
- Python 3.6
- Chainer 4.0
- matplotlib
- tqdm

## Usage
Chainer model is defined with `network.py`
### Train
```
python train_CAE.py -g 0 -e 20
```

### Show result
```
python generate.py
```

### Plot model
First, you have to install `Graphviz`. If you use Anaconda, you can install with next command.
```
conda install graphviz
```

Second, Run `plot_model.py`. Then, `graph.dot` is generated.  

Third, convert dot file to png file with next command.
```
dot -Tpng graph.dot -o graph.png
```

## Model
<img src="https://github.com/takyamamoto/CIFAR-ConvolutionalAutoEncoder-Chainer/blob/master/graph.png" width=20%>

## Result
<img src="https://github.com/takyamamoto/CIFAR-ConvolutionalAutoEncoder-Chainer/blob/master/result.png">
