# Components and licensing

Project licensing is in [LICENSE](LICENSE).

- PyTorch supplies tensor operations, reference calculations and CUDA execution. See its upstream BSD-style license and bundled dependency notices.
- Triton supplies the GPU kernel compiler and language; it is distributed under its upstream MIT license.
- NumPy supplies array manipulation for CSR scheduling; SciPy is used by the evaluation utilities. Both retain their upstream BSD-style licenses.
- AGH supplies the model runtime and native plugin interface. Install AGH separately and retain its upstream Apache-2.0 and component notices when distributing that runtime.

The project contains the operator implementations, native plugin and local evaluation support. It uses each dependency through the interfaces described in the running guide.
