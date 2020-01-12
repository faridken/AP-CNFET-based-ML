# Exploiting Dual-Gate Ambipolar CNFETs for Scalable Machine Learning Classification

## Project Description

Ambipolar carbon nanotube based field-effect transistors (AP-CNFETs) exhibit unique electrical characteristics, such as tri-state operation and bi-directionality, enabling systems with complex and reconfigurable computing. In this paper, AP-CNFETs are
used to design a mixed-signal machine learning (ML) classifier. The classifier is designed in SPICE with feature size of 15 nm
and operates at 250 MHz. The system is demonstrated based on MNIST digit dataset, yielding 90% accuracy and no accuracy
degradation as compared with the classification of this dataset in Python. The system also exhibits lower power consumption
and smaller physical size as compared with the state-of-the-art CMOS and memristor based mixed-signal classifiers.

## Paper
F. Kenarangi, X. Hu, Y. Liu, J.A.C. Incorvia, J.S. Friedman, and I. P.-Vaisband, “Exploiting Dual-Gate
Ambipolar CNFETs for Scalable Machine Learning Classification,” Scientific Reports, 2019 (under review,
[Link](https://arxiv.org/abs/1912.04068)).

## Directory Structure
- data - input data (features and feature weights)
- lib - library of the components used for simulation
- src - simulation files (netlist of the circuit)
- log - results of the simulation

## Verified platform
[Centos 7](https://www.centos.org/)

## Run Simulation

1. Download the repository directly as a zip file or using git clone command:

`git clone https://github.com/faridken/AP-CNFET_based_ML.git`


2. Go to the src directory 

`cd AP-CNFET_based_ML/src`

3. Make sure you have downloaded the files to a directory where you can envoke spectre simulator in Cadence




