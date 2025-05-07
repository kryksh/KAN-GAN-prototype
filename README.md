# KAN-GAN-prototype

The repository contains modifications of methods for generating synthetic tabular data using KAN instead of MLP.

The models folder contains modifications of TVAE and CTGAN models from [here](https://github.com/sdv-dev/CTGAN/tree/main/ctgan/synthesizers), where instead of Linear-layers is used KANLinear-layers from [here](https://github.com/Blealtan/efficient-kan).

The notebook experiment contains an example of data generation using different models, as well as proposed modifications. Experimental studies contain two parts: (1) solving the classification and multi-label classification problem on different synthetic data and comparing metrics; (2) comparing synthetic and real data according to metrics from the [SDMetrics library](https://github.com/sdv-dev/SDMetrics).