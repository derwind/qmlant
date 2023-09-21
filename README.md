# qmlant

QML (Quantum Machine Learning) SDK based on [cuTensorNet](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html).

- Simple Quantum Neural Networks
- Simple Quantum Convolutional Neural Networks
- Simple QAOA (X-Mixer, XY-Mixer)
- Simple VQE

## Installation

```bash
pip install .
```

## Acknowledgements

- [optimizers.py](qmlant/optim/optimizers.py) is adopted almost as is from [dezero/optimizers.py](https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/dezero/optimizers.py).
- [circuit_converter.py](qmlant/visualization/circuit_converter.py) is adopted from [qiskit/qasm2/parse.py](https://github.com/Qiskit/qiskit/blob/main/qiskit/qasm2/parse.py).
