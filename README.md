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

1. **topology-analysis.py**
   
**Input**
- Flattened gate-level Verilog netlist (e.g., `aes.v`)

**Output**
- `cell2pin.pkl`

This script extracts the **topological connectivity** between logic cells and signals from a flattened gate-level netlist. It parses the netlist and constructs a cell-centric representation of the circuit structure.

Although this script is used in the AIPS data preparation pipeline, it is **not specific to AIPS**.
It can be generally applied to:
- Netlist topology extraction
- Structural analysis of digital circuits
- Graph construction for graph neural networks (GNNs)
- Any task requiring cell-level connectivity information

---

The output file `cell2pin.pkl` is a Python dictionary with the following structure:

- **Key**: Cell instance name
- **Value**: A list containing
  1. The cell type
  2. A list of two dictionaries:
     - Input pin–to–net mapping
     - Output pin–to–net mapping

**Example:**

```python
control_U30 [
    'XNOR2X1',
    [
        {'A': 'control_gray_dout[0]', 'B': 'control_gray_dout[3]'},
        {'Y': 'control_n5'}
    ]
]```

### Diffusion Model

This stage implements the diffusion-based power simulation model used in AIPS. The model learns to generate realistic power traces conditioned on VCD traces and cell-level features.

The scripts in this folder include model definition, training procedures,
and sampling (power trace generation).

### Security Evaluation




