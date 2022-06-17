
# Exploding_brains (wotk in progress...)
Particle simulations for exploding brains **in Julia**. This is the successor to the [slowest-particle-simulator-on-earth](https://github.com/ofgulban/slowest-particle-simulator-on-earth). I am going to develop this project further during [OHBM Brainhack 2022](https://ohbm.github.io/hackathon2022/). Previous hackathon project are:
1. [2020 OpenMR Benelux](https://github.com/ohbm/hackathon2020/issues/124)
2. [2020 OHBM Brainhack](https://github.com/OpenMRBenelux/openmrb2020-hackathon/issues/7)

For further details on the particle simulations read this excellent blog post: https://nialltl.neocities.org/articles/mpm_guide.html

<!--(https://github.com/ofgulban/exploding_brains/blob/main/visuals/example-17.gif)-->
<img src="/visuals/example-17.gif" width=256 align="center" />

## Dependencies
| Package | Tested Version |
|---------|----------------|
|   NIfTI |           0.5.6|

## Installation
1. Clone this repository.
2. Navitage into the `wip` folder.
3. Before you try to run any scripts within, chnage the `include(.../core.jl)` into your own system path. E.g. `include(/faruk/git/exploding_brains/wip/core.jl)` to `include(/leonardo/exploding_brains/wip/core.jl)`.
4. Initialize Julia interactive command line interface and run one of the example scripts by `include(./01_move_particles.jl)`

Feel free to open an issue in this repository if something goes wrong. This project is very much **work in progress**.

## Usage
TODO

## Making a gif
TODO

## Support
Please use [GitHub issues](https://github.com/ofgulban/slowest-particle-simulator-on-earth/issues) for questions, or comments.

## License
This project is licensed under [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause).
