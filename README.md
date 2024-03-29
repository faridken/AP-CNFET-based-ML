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
- data - input data
- lib - library of the components used for simulation
- src - simulation files (netlist of the circuit)
- log - log file for debugging and viewing dependencies
- output - results of the simulation 

## Verified platform
[CentOS Linux release 7.6.1810](https://www.centos.org/)

## Software

The simulations are performed using "Cadence (R) Virtuoso (R) Spectre (R) Circuit Simulator"

Version 15.1.0.257 64bit -- 9 Oct 2015 

Copyright (C) 1989-2015 Cadence Design Systems, Inc. All rights reserved worldwide. Cadence, Virtuoso and Spectre are registered trademarks of Cadence Design Systems, Inc. All others are the property of their respective holders.



## Run Simulation

1. Download the repository directly as a zip file or using git clone command:

`git clone https://github.com/faridken/AP-CNFET_based_ML.git`

2. Make sure you have downloaded the files to a directory where you can envoke spectre simulator of Cadence

3. To point out to the correct component models you may need to change the dicretory of the modesl within netlist file (last two lines as attached below)

`ahdl_include "../lib/AP-CNFET/veriloga/veriloga.va"`

`ahdl_include "/opt/IC617/tools/dfII/samples/artist/ahdlLib/analog_mux/veriloga/veriloga.va"`

In this case we read the transistor models from the lib directory and read the analog_mux model from cadence directory (which varies based on your installation directory).

4. Go to the src directory 

`cd AP-CNFET_based_ML/src`

5. Enovoke the simulation using the commad below

`spectre -64 ++aps +mt AP-CNFET.netlist`

6. The simulation results will be stored in "AP-CNFET.raw" file.

7. Simulation results can be analyzed using "Virtuoso Visualisation and Analysis (ViVA)" by envoking the command below

`viva -mode xl`

![alt text](images/waveform.png)

## Simulation Results

The votes/decisions of binary classifiers are sampled with a frequency of 250MHz and are stored in "/output/decisions.csv". Finally the accuracy is obtained based on these decisions using a simple script (i.e., Accuracy.ipynb).      

![alt text](images/acc.png)



