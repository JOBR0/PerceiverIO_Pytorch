# PerceiverIO Pytorch

Adaptation of Deepmind's PerceiverIO model (https://arxiv.org/abs/2103.03206) to Pytorch.
The original jax/haiku code can be found here:
https://github.com/deepmind/deepmind-research/tree/master/perceiver


The implementation covers the following example tasks:

* Masked language modelling
* Image classification
* Multi-modal video auto-encoding
* Optical flow estimation

## Usage

`jj`

### Input preprocessors
Input preprocessors take the raw input data and preprocess it so that it can be queried by the 
first cross-attention. This can be something like creating patches from an image. Usually positional encodings are
added by the preprocessor. Instead of using a preprocessor, the inputs can also be processed externally.

### Output postprocessors
Output postprocessors take the final output of the perceiver and process it to the final format.

### Output queries
Ouput queries create the features that are used to query the final latent representation of the perceiver to produce the output.
They are passed the preprocessed input so that they can use it if desired. They also usually add positional encodings.

The decoder cross-attends to the processed latent features to produce the output.

There are several input_prepprocesors available for different tasks. 

The preprocessor transforms the input before it is attended to and usually adds positional information.
The postprocessor transforms the output of the decoder to get the final output.

###Multimodal
To process multiple modalities at once, a dictionary with a mapping from modality to the module can be used for the input_preprocessors, output_postprocessors and the output_queries (see perceiver_io/multimodal_perceiver.py).
To make the different inputs compatible with each other, they are padded to the same channel size with trainable parameters.

## Checkpoints

The haiku checkpoints from the official deepmind repository have been converted to PyTorch checkpoints and can be downloaded from <a href= "https://drive.google.com/drive/folders/1ks00isq02LaACvE405dIwZqUfWxC0irV?usp=sharing">google-drive</a>.
The pytorch checkpoints should be placed in the 'pytorch_checkpoints' folder so that the example code can find them.







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