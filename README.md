# PerceiverIO Pytorch

Adaptation of Deepmind's PerceiverIO model (https://arxiv.org/abs/2103.03206) to Pytorch.
The original jax/haiku code can be found here:
https://github.com/deepmind/deepmind-research/tree/master/perceiver


## Installation

- Clone the repository:
```bash
git clone https://github.com/JOBR0/PerceiverIO_Pytorch
cd PerceiverIO_Pytorch
```

- Create a virtual environment and activate it:
```bash
python -m venv perceiverEnv
source perceiverEnv/bin/activate
```

- Install pytorch following the official instructions:
https://pytorch.org/get-started/locally/

- Install other required packages from requirements.txt:
```bash
pip install -r requirements.txt
```

## Examples
The implementation covers the following example tasks for which pretrained models are available:

* Masked language modelling (example_language.py)
* Image classification (example_img_classify.py)
* Multi-modal video auto-encoding (example_multimodal.py)
* Optical flow estimation (example_opt_flow.py)

### Pretrained models

The haiku checkpoints from the official deepmind repository have been converted to PyTorch checkpoints and can be downloaded from <a href= "https://drive.google.com/drive/folders/1ks00isq02LaACvE405dIwZqUfWxC0irV?usp=sharing">google-drive</a>.
The pytorch checkpoints should be placed in the 'pytorch_checkpoints' folder so that the example code can find them.

## Usage

To create a new preceiver IO for a custom task, the Perceiver class in perceiver_io/perceiver.py is used.


```python
class Perceiver(nn.Module):
    """The Perceiver: a scalable, fully attentional architecture.
    Args:
        num_blocks (int): Number of times the block is applied with shared weights. Default: 8
        num_self_attends_per_block (int): Number of self-attentions in the block. Default: 6,
        num_latents: (int): Number of latent vectors. Default 512,
        num_latent_channels (int): Number of channels for the latent vectors. Default: 1024,
        final_project (bool): Whether to apply a linear layer to the outputs before the post-processors. Default: True,
        final_project_out_channels (int): Number of output channels for the final projection layer. Default: None,
        perceiver_encoder_kwargs (Dict): Additional arguments for the perceiver encoder class. Default: {},
        perceiver_decoder_kwargs (Dict): Additional arguments for the perceiver decoder class. Default: {},
        input_preprocessors (dict / nn.Module): Optional input preprocessors. 1 or none for each modality. Default: None,
        output_postprocessors (dict / nn.Module): Optional output postprocessors. 1 or none for each modality. Default: None,
        output_queries (dict / nn.Module): Modules that create the output queries. 1 for each modality. Default: None,
        output_query_padding_channels (int): Number of learnable features channels that are added to the output queries. Default: 0,
        input_padding_channels (int): Number of learnable features channels that are added to the preprocessed inputs. Default: 0,
        input_channels (dict, int): = The number of input channels need to be specified if NO preprocessor is used. Otherwise,
                                    the number will be inferred from the preprocessor. Default: None,
        input_mask_probs (dict): Probability with which each input modality will be masked out. Default None,
    """
```



### Input preprocessors (optional)
Input preprocessors take the raw input data and preprocess it so that it can be queried by the 
first cross-attention. This can be e.g. something like creating patches from an image. Usually positional encodings are
incorporated by the preprocessor. Instead of using a preprocessor, the inputs can also be processed manually.

### Output postprocessors (optional)
Output postprocessors take the final output of the perceiver and process it to obtain the desired output format.

### Output queries
Ouput queries create the features that are used to query the final latent representation of the perceiver to produce the output.
They obtain the preprocessed input as an argument so that they can use it if desired. They also usually incorporate positional encodings.


###Multiple modalities
To process multiple modalities at once, a dictionary with a mapping from modality to the module can be used for the input_preprocessors, output_postprocessors and the output_queries (see perceiver_io/multimodal_perceiver.py).
To make the different inputs compatible with each other, they are padded to the same channel size with trainable parameters.









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