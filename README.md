# AIPS: AI-Based Power Simulation for Pre-silicon Side-Channel Security Evaluation

This directory contains the design files corresponding to the CHES 2026 paper "AIPS: AI-Based Power Simulation for Pre-silicon Side-Channel Security Evaluation" by Ya Gao, Haocheng Ma, Tanchen Zhang, Jiaji He, Yiqiang Zhao, Mirjana Stojilovic, Yier Jin.

---

## Overview

This repository uses the **AES circuit** as a representative example to demonstrate the complete AIPS workflow.

These components correspond to the three major stages of the AIPS pipeline:

- **Data Preparation**
- **Diffusion Model**
- **Security Evaluation**

---

## Table of Contents

- [System Configuration Used for Experiments](#system-configuration-used-for-ex)
- [Python requirement](#python-requirement)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)


## Usage

### Data Preparation

This stage prepares the structural and activity-related information required for AIPS. It processes the gate-level netlist and signal activities to extract circuit topology, cell-level features, and VCD traces.

#### topology-analysis.py

```
This script extracts the **topological connectivity** between logic cells and signals
from a flattened gate-level netlist. It parses the netlist and constructs a
cell-centric representation of the circuit structure.

Input:
- Flattened gate-level Verilog netlist (e.g., aes.v)
Output:
- cell2pin.pkl

The output file `cell2pin.pkl` is a Python dictionary with the following structure:
- Key:   Cell instance name
- Value: A list containing
  1. The cell type
  2. A list of two dictionaries:
     - Input pin–to–net mapping
     - Output pin–to–net mapping
```

**Example**

```python
control_U30 [
    'XNOR2X1',
    [
        {'A': 'control_gray_dout[0]', 'B': 'control_gray_dout[3]'},
        {'Y': 'control_n5'}
    ]
]
```

#### Lib_parser.py

This script parses a Liberty file and extracts **cell leakage power** together with **internal power** information. For each cell, it summarizes the `internal_power` tables by taking the **median** value of the `values(...)` entries.

**Input**
- Liberty `.lib` file (e.g., `fast.lib`)

**Output**
- `cell_power.pkl`

The output is stored as a Python dictionary:

- **Key**: Cell name (e.g., `XNOR2X1`)
- **Value**: A list of floats, where:
  1. The **first element** is `cell_leakage_power`
  2. The remaining elements are the extracted internal-power summary values

**Example**

```python
XNOR2X1: [355.859064, 0.051917, 0.084137, 0.080041, 0.067042, 0.09906, 0.14078, 0.133293, 0.111911]
```

### Diffusion Model

This stage implements the diffusion-based power simulation model used in AIPS. The model learns to generate realistic power traces conditioned on VCD traces and cell-level features.

The scripts in this folder include model definition, training procedures,
and sampling (power trace generation).

### Security Evaluation




