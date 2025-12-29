# AIPS: AI-Based Power Simulation for Pre-silicon Side-Channel Security Evaluation

This directory contains the design files corresponding to the CHES 2026 paper "AIPS: AI-Based Power Simulation for Pre-silicon Side-Channel Security Evaluation" by Ya Gao, Haocheng Ma, Tanchen Zhang, Jiaji He, Yiqiang Zhao, Mirjana Stojilovic, Yier Jin.

---

## Overview

This repository uses the **AES circuit** as a representative example to demonstrate the complete AIPS workflow.

These components correspond to the three major stages of the AIPS pipeline:

- **Data Preparation**
- **Diffusion Model**
- **Security Evaluation**

> **Note:**  
> Please preserve the directory structure provided in this repository, as the scripts assume fixed relative paths between different components.

---

## Table of Contents

- [System Configuration Used for Experiments](#system-configuration-used-for-ex)
- [Python requirement](#python-requirement)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)


## Usage

### Data Preparation

This stage prepares the structural and activity-related information required for diffusion-based power simulation. It processes the gate-level netlist and signal activity traces to construct
topology-aware inputs for the AIPS model. The scripts in this folder are responsible for extracting circuit topology,
cell-level features, and aligned activity representations.


### Diffusion Model

### Security Evaluation

### Data Preparation



