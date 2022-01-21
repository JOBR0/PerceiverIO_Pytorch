# PerceiverIO Pytorch

Adaptation of Deepmind's PerceiverIO model (https://arxiv.org/abs/2103.03206) to Pytorch.



The implementation stays close to the original implementation and covers the following tasks:

* Masked language modelling
* Image classification
* Multi-Modal Video Classification
* Optical Flow estimation

## Usage

The perceiver IO has an encoder and a decoder. The encoder cross-attends to the inputs and transforms the attended features through the of self-attention blocks.

The decoder cross-attends to the processed latent features to produce the output.



## Checkpoints

The haiku checkpoints from the official deepmind repository have been converted to PyTorch checkpoints and can be downloaded from <a href= "https://drive.google.com/drive/folders/1ks00isq02LaACvE405dIwZqUfWxC0irV?usp=sharing">google-drive</a>.
The pytorch checkpoints should be placed in the 'pytorch_checkpoints' folder.

There are several different input_prepprocesors available for different tasks. 

The preprocessor transforms the input before it is attended to and usually adds positional information.
The postprocessor transforms the output of the decoder to get the final output.


The original haiku-files can also be loaded, they require, however, the installation of haiku.
TODO check if jax is needed.





## Citations

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver IO: A General Architecture for Structured Inputs & Outputs},
    author  = {Andrew Jaegle and Sebastian Borgeaud and Jean-Baptiste Alayrac and Carl Doersch and Catalin Ionescu and David Ding and Skanda Koppula and Andrew Brock and Evan Shelhamer and Olivier Hénaff and Matthew M. Botvinick and Andrew Zisserman and Oriol Vinyals and João Carreira},
    year    = {2021},
    eprint  = {2107.14795},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```