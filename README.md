# AerAudioViz

`AerAudioViz` is a package for creating videos with dynamic visual effects based on the characteristics of an audio file.

The package includes tools to enable the following:
* Extracting time series features from an audio file in the form of a `pandas.DataFrame`
* Adding additional features and modifying existing features
* Mapping time series features to video effects and feature values to the effect parameters
* Generating video based on a base image and the defined video effect mappings

## Installation

### Using `poetry`
Include as a dependency in your project by using `poetry add git+ssh://git@github.com:Aerodactylus/AerAudioViz.git`.

To install with `poetry` for local development, clone the repository and run `poetry install --all-extras` in the root directory of the project.

## Usage Example
`aeraudioviz` was used to create the video for the album *Jetsam Dreams* by *Aerodactylus* which you can view [here on YouTube](https://youtu.be/vkKwFKCnxnw). 

The code used to generate that video can be found in the the IPython Notebooks in the [*Aerodactylus/JetsamDreams* GitHub repository](https://github.com/Aerodactylus/JetsamDreamsVideo/). These IPython Notebooks provide a full demonstration of the `aeraudioviz` workflow. 
