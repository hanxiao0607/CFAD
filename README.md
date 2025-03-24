[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/hanxiao0607/CFAD/blob/main/LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)

# CFAD: Achieving <u>C</u>ounterfactual <u>F</u>airness for <u>A</u>nomaly <u>D</u>etection

A Pytorch implementation of [CFAD](https://arxiv.org/abs/2303.02318).

## Configuration
- Ubuntu 20.04
- NVIDIA driver 470.74
- CUDA 11.1
- Python 3.9.7
- PyTorch 1.9.1

##  Hardware
- Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
- 64 GB Memory
- NVIDIA GeForce RTX 2080 Ti


## Installation
This code requires the packages listed in requirements.txt.
A virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate
```
Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

## Instructions
Clone the template project, replacing ``my-project`` with the name of the project you are creating:

        git clone https://github.com/hanxiao0607/CFAD.git my-project
        cd my-project

Run and test:

        python3 CFAD_adult.py
        or
        python3 CFAD_compas.py
        or
        python3 CFAD_synthetic.py

## Citation
```
@inproceedings{han2023achieving,
  title={Achieving Counterfactual Fairness for Anomaly Detection},
  author={Han, Xiao and Zhang, Lu and Wu, Yongkai and Yuan, Shuhan},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={55--66},
  year={2023},
  organization={Springer}
}
```
