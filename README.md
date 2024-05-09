# Universal Spatial Audio Transcoder (USAT)
This code consists of a universal transcoder tool. It generates a
psychoacoustically motivated transcoding matrix to transform from any input
format to any other given format or speaker layout, maximizing the preservation of spatial information.

## Paper
See more details about the theoretical foundation and mathematical procedures behind this code in the paper *Universal Spatial Audio Transcoder* ([preprint](https://arxiv.org/abs/2405.04471)).

> A. Sagasti, D. Scaini, and D. Arteaga, “Universal spatial audio transcoder,” 2024. Accepted for
presentation at the *AES 156th Convention*, Madrid, Spain. https://arxiv.org/abs/2405.04471


```
@misc{usat_2024,
      title = {Universal Spatial Audio Transcoder}, 
      author = {Amaia Sagasti and Davide Scaini and Daniel Arteaga},
      year = {2024},
      eprint = {2405.04471},
      archivePrefix = {arXiv},
      primaryClass = {cs.SD},
      note = {Accepted for presentation at the AES 156th Convention, Madrid, Spain}
}
```

In the folder `paper/saved_results` you can find all data and plots corresponding to the results in the paper.

## Getting started

### Pre-requisites

It is recommended Python 3.8 or newer. 

You need the [GEOS](https://libgeos.org/) package. To install it on mac with Homebrew, you can:
```
brew install geos
```

One way to use `universal_transcoder` is to create a virtual environment 
```
python3.8 -m venv venv
source ./venv/bin/activate
```
and install `universal_transcoder` with:
```
pip install -e .
```
Alternatively, you can also install the dependencies from the `requirements.txt` file:
```
pip install -r requirements.txt
```

### Run code
To try out this code, you can run the 4 scripts prepared with the examples shown in the paper:
`ex1_5OAto704.py`, `ex2_704to5OA.py`, `ex3_502to301irr.py` and `ex4_ObjectTo50.py` (found in the `paper` folder).

The code corresponding to the USAT algorithms is located in the folder `universal_transcoder`.

### Run tests
```
python -m pytest tests 
```

## How it works

USAT is an algorithm capable of calculating, through an optimization based on some psychoacoustic effects, the optimised transcoding matrix `T_optimised` that transcodes a defined input format of $M$ channels to an, also defined, output format of $N$ channels.

<img src="USAT-schema.png" width="100%"/>

A dictionary like the one below is passed as input to the function `optimize()` inside `calculations/optimization.py`, which generates as output the optimized transcoding matrix. 


```
        dictionary = {
            "input_matrix_optimization": input_matrix_optimization, # Input matrix that encodes in input format (LxM) **
            "cloud_optimization": cloud_optimization,               # Cloud of points sampling the sphere (L) **
            "output_layout": output_layout,                         # Output (real/virtual) layout of speakers to decode(P) **
            "Dspk": Dspk,                                           # Decoding matrix from output format to layout of speakers (PxN) ***
            "coefficients": {                                       # List of coefficients to the cost function **
                "energy": 5,
                "radial_intensity": 2,
                "transverse_intensity": 1,
                "pressure": 0,
                "radial_velocity": 0,
                "transverse_velocity": 0,
                "in_phase_quad": 10000,
                "symmetry_quad": 0,
                "in_phase_lin": 0,
                "symmetry_lin": 0.0,
                "total_gains_lin": 0,
                "total_gains_quad": 0,
                "sparsity_quad": 0.01,
                "sparsity_lin": 0.001,
            },
            "directional_weights": 1,                               # Weights to directions sampling the sphere (1xL)
            "show_results": show_results,                           # Flag to show results **
            "save_results": save_results,                           # Flag to save results **
            "results_file_name": "ex3_50to301irr_USAT",             # Name of folder to save results 
            "input_matrix_plots": input_matrix_plots,               # Auxiliary matrix that encodes in input format (L'xM)
            "cloud_plots": cloud_plots,                             # Auxiliary cloud of points sampling the sphere for plotting (L')
            "T_initial": T_initial,                                 # Starting point of optimization 
        }
```

** mandatory inputs

*** mandatory only if output format is layout independent, like Ambisonics

This dictionary is passed as input to the function `optimize()` inside the module `universal_transcoder/calculations/optimization.py. The following sections will explain in more detail each entry of the dictionary.

### The clouds - `cloud_optimization` and `cloud_plots`
Variables containing set of points sampling the sphere.

These variables are formatted as the class `MyCoordinates`, which is a subclass of `pyfar.Coordinates`. They can be generated using the functions inside `auxiliars/get_cloud_points.py`. 

- The data saved in key `"cloud_optimization"` corresponds to the set of points ($L$ points) sampling the sphere in which the optimization is desired. If the output format or speaker layout is 2D, the most appropiate cloud would be a 2D set of points, which can be generated using `get_equi_circumference_points()` (from `auxiliars/get_cloud_points.py`). On the other hand, if the output format or layout to which we aim to decode is 3D, it would be more appropiate to generate a 3D set of sampling points (`get_sphere_points()`, `get_equi_t_design_points()`, `get_equi_fibonacci_sphere_points()`, `get_all_sphere_points()` from `auxiliars/get_cloud_points.py`). It is possible to use any other function that generates points in the same format. For the opimization, it is recommended to have a set of points that are equally distributed in terms of energy across the sphere.

- For the case of key `"cloud_plots"`, this set of points ($L'$ points) corresponds to the sampling directions of the sphere to be shown in the plots if either `"save_results"` or `"show_results"` keys are active. If `"cloud_plots"` is not defined but `"save_results"` or `"show_results"` are active, the program will use `"cloud_optimization"` for the plots. Similarly to the case above, depending on the dimensions of the output layout, `"cloud_plots"` should be set accordingly.


```
from auxiliars.get_cloud_points import get_equi_circumference_points,get_sphere_points,get_all_sphere_points
cloud_2D = get_equi_circumference_points(10)
cloud_3D = get_sphere_points(8)
```


### The input matrices - `input_matrix_optimization` and `input_matrix_plots`

Variables containing the encoding gains that encode each direction given in a cloud of points in a specific audio format. This constitutes one of the main inputs to the program, due to the fact it provides the information about the input format that we aim to decode to a speaker layout.

These variables are formatted as `numpy.Array`. There are auxiliary functions to compute them for the most common spatial audio formats inside `auxiliars/get_input_channels.m`.

- The data stored in key `"input_matrix_optimization"` corresponds to the encoding gains that encode the set of $L$ directions in the cloud, into the input audio format of $M$ channels. Array of size $L\times M$.

- The data stored in key `"input_matrix_plots"` corresponds to the encoding gains that encode the set of L' directions in the cloud, into the input audio format of $M$ channels. Array of size $L'\times M$. This is only used for plotting. If `cloud_plots` is not defined but `save_results` or `show_results` are active, the program will use `cloud_optimization` and `input_matrix_optimization` for the plotting.

NOTE: Both of these variables must be generated with the same encoder: different set of points but same encoder.

```
from auxiliars.get_input_channels import get_input_channels_ambisonics,

order=1

#Clouds
cloud_optimization = get_sphere_points(8, False)
cloud_plots = get_all_sphere_points(5, False)

#Input matrices
input_matrix_optimization = get_input_channels_ambisonics(cloud_optimization, order)
input_matrix_plots = get_input_channels_ambisonics(cloud_plots, order)

```

### Output layout - `output_layout`
Variable containing the set of directions where the speakers (real, in case of decoding to a layout of speakers, or virtual, in case of transcoding to a layout independent format, like Ambisonics) to which we aim to decode are located, ($P$ speakers).

These variables are formatted as `MyCoordinates`. They are generated manually using the methods of MyCoordinates. For example, the method mult_points() generates the variable `layout`receiving as input an array of size $N\times 3$, in which columns correspond respectively to the azimuth, elevation, and radius of the speakers.

```
layout = MyCoordinates.mult_points(
    np.array(
        [
            (-120, 0, 1),
            (-30, 0, 1),
            (0, 0, 1),
            (30, 0, 1),
            (120, 0, 1),
        ]
    )
)
```

### Coefficients - `coefficients`
These variables establish the different weights given to the different terms of the cost function, consequently giving more or less importance to the different psychoacoustical effects. Set 0 if variable is inactive.

### Directional weights - `directional_weights`
These variables establish the different weights given to the different directions of the cloud of points, providing the possibility of giving more or less importance in the optimization to certain zones (for example those points close to speakers in the output), at expense of other zones.

These variables are formatted as a `numpy.array` of size $1xL$. Set 1 if directional weighting is unneeded.

### Results - `show_results` , save_results and `results_file_name`

If key `show_results` is active, the system will show plots and print some logs through terminal.

If key `save_results` is active, the system will store all the resulting plots and logs in folder `/saved_results` inside a new folder called as set in key `results_file_name`.

### Initial transcoding matrix - `T_initial`
This variable is optional and defines the initial point of the optimization. Its size must be $N \times M$. In case it is not  passed as input, the algorithm generates a random matrix of size $N \times M$. 

### Decoding-to-speakers matrix - `Dspk`
This variable represents the decoding matrix of shape $P \times N$ needed to decode the output format to the defined `output_layout` of speakers. It is not needed when the output format already constitutes a set of N speakers (`Dspk=1, N=P`). However, it is mandatory when the desired output format is a speaker-independent format, like Ambisonics. 

In the code, it is provided the function `get_ambisonics_decoder_matrix()` in `universal_transcoder/auxiliars/get_decoder_matrices.py`, as implemented in `ex2_704to5OA.py`, an example of transcoding to Ambisonics (output format = Ambisonics 5th Order)

```
Dspk = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
```
