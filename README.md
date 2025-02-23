# Bridging Classical and Quantum Machine Learning 

[![arXiv](https://img.shields.io/badge/arXiv-2311.13810-b31b1b.svg)](https://arxiv.org/abs/2311.13810)

Welcome to **Bridging Classical and Quantum Machine Learning**, a repository that combines **Qiskit basics** with **PyTorch-based** quantum–classical hybrid experiments. This project introduces how to leverage knowledge distillation to transfer capabilities from classical neural networks to resource-constrained quantum neural networks. In addition, the repository provides **Qiskit** fundamentals through the `Qiskit Basics.ipynb` notebook to help you get started with quantum computing concepts and simulations.

---

## Table of Contents

1. [Motivation](#motivation)  
2. [Architecture and Approach](#architecture-and-approach)  
   - [Knowledge Distillation Method](#knowledge-distillation-method)  
   - [Dimensionality Techniques](#dimensionality-techniques)  
3. [Qiskit Basics](#qiskit-basics)  
4. [Project Structure](#project-structure)  
5. [Installation and Setup](#installation-and-setup)  
6. [Usage](#usage)  
7. [Experiments](#experiments)  
   - [MNIST](#mnist)  
   - [FashionMNIST](#fashionmnist)  
8. [Results](#results)  
9. [Citation](#citation)  
10. [License](#license)  

---

## Motivation

Quantum neural networks (QNNs) promise a new frontier in machine learning, potentially surpassing classical models when they have a similar number of parameters. However, challenges such as limited qubit availability, decoherence, and hardware noise make large-scale training of quantum models difficult. 

This project addresses these obstacles by **transferring knowledge** from large, well-trained **classical** neural networks to **quantum** student models. Through knowledge distillation, we bypass the requirement of training large quantum networks, making it more feasible to reap the benefits of quantum computing for tasks like image classification.

---

## Architecture and Approach

### Knowledge Distillation Method

In our paper, “Bridging Classical and Quantum Machine Learning: Knowledge Transfer From Classical to Quantum Neural Networks Using Knowledge Distillation,” we show how a **frozen classical CNN** (teacher) transfers knowledge to a **smaller quantum network** (student). Below is a high-level diagram illustrating the overall training scheme:

![Quantum-Classical Knowledge Distillation Architecture](ss1.png)

We compare our **classical-to-quantum** knowledge distillation approach to (1) classical-to-classical and (2) quantum-to-quantum distillation methods:

![Motivation and Different Distillation Strategies](ss2.png)

1. **Teacher Model (Classical, Frozen)**  
   - Typically a large CNN (e.g., LeNet, AlexNet).  
2. **Student Model (Quantum, Trainable)**  
   - A QNN with significantly fewer trainable parameters.  
3. **Distillation Loss**  
   - Minimizes KL divergence between teacher’s and student’s output logits.

### Dimensionality Techniques

Because quantum circuits can only handle a limited number of qubits, we must **reduce input dimensionality** before encoding data into a quantum circuit. We experiment with several strategies:

![Dimensionality Reduction for Quantum Processing](ss3.png)

1. **Fully Connected (Flatten + FC):**  
   - Flatten the image, project to \(2^Q\) features for \(Q\) qubits.  
2. **Average/Max Pooling:**  
   - Divide the image into \(2^Q\) regions and pool values.  
3. **Center Crop:**  
   - Crop the central \(N \times N\) patch with \(N^2 = 2^Q\).  
4. **PCA:**  
   - Use Principal Component Analysis to extract \(2^Q\) components.

Below is an example of the **error rates** for different dimensionality strategies across MNIST, FashionMNIST, and CIFAR10 for 4-qubit and 8-qubit QNNs:

![Error Comparison of Methods](ss4.png)

---

## Qiskit Basics

Alongside our quantum–classical experiments, this repository includes a Jupyter notebook named **`Qiskit Basics.ipynb`**, where you can learn:

- How to create quantum circuits using Qiskit  
- Fundamental quantum operations (gates, measurements)  
- Running quantum circuits on simulators  
- (Optionally) deploying circuits on real quantum hardware

This serves as a gentle introduction to quantum computing concepts that underpin the experiments in the **student** QNN.

---

## Project Structure

```
.
├── Qiskit Basics.ipynb                # Fundamentals of Qiskit
├── MNIST Experiments
│   ├── Teachers                       # Classical CNN teacher notebooks
│   ├── Baseline students             # Quantum students without distillation
│   └── Distillation on students      # Quantum students with knowledge distillation
├── FashionMNIST Experiments
│   ├── Teachers
│   ├── Baseline students
│   └── Distillation on students
├── ss1.png                            # Architecture diagram
├── ss2.png                            # Motivation & method comparison diagram
├── ss3.png                            # Dimensionality reduction methods
├── ss4.png                            # Error comparison chart
└── README.md                          # You are here!
```

---

## Installation and Setup

To run the experiments and Qiskit tutorials, you need:

- **Python 3.7 or higher**
- **PyTorch 1.8 or higher**
- **Qiskit 0.25 or higher**  
- **TorchQuantum 0.1.0 or higher**
- **Jupyter Notebook or Jupyter Lab**

### Quick Installation

```bash
pip install torch            # For PyTorch
pip install qiskit           # For Qiskit
pip install torchquantum     # For TorchQuantum
pip install jupyter          # For notebooks
```

### Optional: Virtual Environment

It’s recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
Then proceed with the installations within your virtual environment.

---

## Usage

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your_username/quantum-machine-learning.git
   cd quantum-machine-learning
   ```

2. **Launch Jupyter**  
   ```bash
   jupyter notebook  # or jupyter lab
   ```
3. **Open Notebooks**  
   - `Qiskit Basics.ipynb` (to learn essential quantum operations).  
   - `MNIST Experiments` or `FashionMNIST Experiments` notebooks (for classical teacher training, baseline quantum models, and knowledge-distilled quantum models).

If you prefer **Google Colab**, simply upload the notebooks and select the appropriate runtime.

---

## Experiments

### MNIST

1. **Teachers**  
   - Train classical CNNs (LeNet, etc.) on MNIST.  
2. **Baseline Students**  
   - Train quantum models without using distillation.  
3. **Distillation on Students**  
   - Transfer knowledge from the frozen classical teacher to the quantum student.

### FashionMNIST

Follow the same process as MNIST but with FashionMNIST data:
1. **Teachers**  
2. **Baseline Students**  
3. **Distillation on Students**  

These steps highlight how knowledge distillation improves QNN accuracy across different datasets.

---

## Results

- **MNIST**: Average quantum model accuracy improves by **0.80%** with distillation.  
- **FashionMNIST**: Average quantum model accuracy improves by **5.40%** with distillation.  
- **CIFAR10** (in some ablation studies): Also shows enhancement, although absolute performance is more challenging due to dataset complexity.

Refer to the **Error Comparison** chart (`ss4.png`) for a visual summary of various dimensionality reduction strategies and 4-qubit/8-qubit experiments.

---

## Citation

If you find this repository useful in your research, please consider citing our work:

```bibtex
@article{hasan2023bridging,
  title={Bridging Classical and Quantum Machine Learning: Knowledge Transfer From Classical to Quantum Neural Networks Using Knowledge Distillation},
  author={Hasan, Mohammad Junayed and Mahdy, MRC},
  journal={arXiv preprint arXiv:2311.13810},
  year={2023}
}
```

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  

**Happy Quantum Coding!** If you have any questions or suggestions, feel free to open an issue or submit a pull request.

&copy; 2025 [Mohammad Junayed Hasan](https://www.linkedin.com/in/mjhasan21/)  

---
