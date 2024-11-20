# PartIIProject

This repository contains the development, simulation, and evaluation of a quantum error correction (QEC) compiler tailored for Quantum Charge-Coupled Devices (QCCD). The project aims to map larger QEC circuits to large-scale QCCD systems, exploring how variations in QCCD architectural parameters and QEC code parameters impact the system's ability to execute active error correction effectively.

## Usage

The repository supports the following primary workflows:

1. **QEC Compiler Development**: The source code in `src/compiler/` provides mapping algorithms for various QEC codes to QCCD hardware. Modifications can be made to these algorithms to test new optimisation techniques or code mappings.

2. **Simulation and Testing**: The simulator in `src/simulator/` allows for the evaluation of compiled QEC circuits, including calculation of logical error rate based on realistic noise models.

3. **Experimentation**: Jupyter notebooks in `experiments/` enable systematic testing of the impact of architectural and QEC code parameters on error correction performance. 

4. **Documentation and Reporting**: The `docs/` directory provides resources for understanding the project, while experimental data reported in the results evaluation of the project are held in `results/`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The development of this project was conducted as a Part II project for the Computer Science Tripos at the University of Cambridge.

- Special thanks to Dr. Prakash Murali for guidance and support throughout the project.

## Contact

For questions or further information, please contact Scott Jones at scott.jones9336@gmail.com.

## Related Work

This project is informed by several key studies in the field of quantum error correction and QCCD architectures:

- **"Transversality and lattice surgery: Exploring realistic routes toward coupled logical qubits with trapped-ion quantum processors"** by Gutiérrez, M., Müller, M., and Bermúdez, A. (2019). This paper explores methods for coupling logical qubits in trapped-ion quantum processors, providing insights into the implementation of logical qubits in QCCD systems. [DOI: 10.1103/PhysRevA.99.022330](https://link.aps.org/doi/10.1103/PhysRevA.99.022330)

- **"Architecting Noisy Intermediate-Scale Trapped Ion Quantum Computers"** by Murali, P., Debroy, D. M., Brown, K. R., and Martonosi, M. (2020). This work discusses the design and challenges of building intermediate-scale quantum computers using trapped ions, offering valuable perspectives on QCCD system architectures. [arXiv:2004.04706](https://arxiv.org/abs/2004.04706)

- **"Experiments with the 4D Surface Code on a QCCD Quantum Computer"** by Berthusen, N., Dreiling, J., Foltz, C., Gaebler, J. P., Gatterman, T. M., Gresh, D., Hewitt, N., Mills, M., Moses, S. A., Neyenhuis, B., Siegfried, P., and Hayes, D. (2024). This recent study presents experimental demonstrations of the 4D surface code on a QCCD quantum computer, providing empirical data on error correction performance in QCCD systems. [arXiv:2408.08865](https://arxiv.org/abs/2408.08865)

- **"Stim: a fast stabilizer circuit simulator"** by Gidney, C. (2021). This paper introduces Stim, a simulator for stabilizer circuits, which is utilised in this project for simulating QEC circuits on QCCD systems. [Quantum 5, 497 (2021)](https://doi.org/10.22331/q-2021-07-06-497)

These studies have significantly influenced the development and methodologies employed in this project. 
