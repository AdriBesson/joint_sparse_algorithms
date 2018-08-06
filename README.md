# Joint Sparsity With Partially Known Support and Application to Ultrasound Imaging 
[Ecole Polytechnique Fédérale de Lausanne (EPFL)]: http://www.epfl.ch/
[Signal Processing Laboratory (LTS5)]: http://lts5www.epfl.ch
[Institute of Sensors, Signals and Systems]: https://www.hw.ac.uk/schools/engineering-physical-sciences/institutes/sensors-signals-systems/basp.htm
[Heriot-Watt University]:https://www.hw.ac.uk/
[paper]:https://infoscience.epfl.ch/record/229453/files/IUS2017_USSR_An_UltraSound_Sparse_Regularization_Framework.pdf

Adrien Besson<sup>1</sup>, Dimitris Perdios<sup>1</sup>, Yves Wiaux<sup>2</sup> and Jean-Philippe Thiran<sup>1,3</sup>

<sup>1</sup>[Signal Processing Laboratory (LTS5)], [Ecole Polytechnique Fédérale de Lausanne (EPFL)], Switzerland

<sup>2</sup>[Institute of Sensors, Signals and Systems], [Heriot-Watt University] , UK

<sup>3</sup>Department of Radiology, University Hospital Center (CHUV), Switzerland

Supporting code of this [paper], submitted to IEEE Signal Processing Letters

## Abstract
We study the performance of joint-sparse algorithms when part of the unknown signal support is known. 
We demonstrate that such a prior knowledge is advantageous compared to rank-aware algorithms when the size of the known support is higher than the rank of the measurements.
We suggest extensions of several joint-sparse recovery algorithms, e.g. multiple signal classification, rank-aware orthogonal recursive matching pursuit and simultaneous normalized iterative hard thresholding. 

We describe a direct application of the proposed methods for compressive multiplexing of ultrasound (US) signals. 
The technique exploits the compressive multiplexer architecture for signal compression and relies on joint-sparsity of US signals in the frequency domain for signal reconstruction.
Due to piezo-electric properties of transducer elements, accurate prior knowledge of the frequency support of US signals is available and can be used in joint-sparse algorithms.

We validate the proposed methods on numerical experiments and show their superiority against state-of-the-art approaches in rank-defective cases.
We also demonstrate that the techniques lead to a significant increase of the image quality on *in vivo* carotid images compared to reconstruction without known support.

## Requirements
  * Python >=3.6 (Code tested with default Python 3.6.2 and Anaconda 4.3.21) 
  * git

## Installation
1. Clone the repository

    ```bash
    git clone  https://github.com/AdriBesson/spl2018_joint_sparse.git
    ```

1. Enter in the `spl2018_joint_sparse` folder

    ```bash
    cd joint_sparse_algorithms
    ```
    
1. (Optional) Create a dedicated Python environment

    * Using Anaconda:

      ```bash
      conda create -n spl2018_joint_sparse python=3.6
      source activate spl2018_joint_sparse
      ```

    * Using `pyenv`:

      ```bash
      pyvenv /path/to/new/virtual/env . /path/to/new/virtual/env/bin/activate
      ```

1. Install Python dependencies from `python_requirements.txt`. Depending on your installation, `pip` may refer to Python 2 (you can verify with `pip -V`). In that case, use `pip3` instead of `pip`.

    ```bash
    pip install --upgrade pip
    pip install -r python_requirements.txt
    ```
    
## Usage
1. `numerical_experiments.py` provides a script to run the experiments to generate Figures 2-a to 2-f 
1. `example_theorem2.pynb` is an explicative notebook that illustrates the benefit of partial known support for rank-aware methods (Theorem 2)
        
## Contact
 Adrien Besson (adrien.besson@epfl.ch)
 
## License
TBD
