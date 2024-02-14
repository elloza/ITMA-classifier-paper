# ITMA-classifier-paper
Datasets and Demo for research paper: Identifying Irish traditional music genres using latent audio representations

# Demo notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elloza/ITMA-classifier-paper/blob/main/demo/DemoTool.ipynb)

## Employed Models:

* Jukemir: https://github.com/p-lambda/jukemir
* Mule: https://github.com/PandoraMedia/music-audio-representations
* MERT: https://github.com/yizhilll/MERT

# Dataset description

The resulting dataset is formed by a main file in CSV format in which the features of all the files are displayed. These features are the following:

- **Name**: it contains the name or title of the tune.

- **Source**: determines the source from which the tune was extracted, ITMA or TheSession.

- **URL**: web link to the tune where you can see more information about it.

- **Genre**: musical genre of the song.

- **Filename**: the name of the file; this name is used in each of the formats in which the dataset is provided (ABC, MusicXML, MIDI, and WAV).

- **Embedding\_jukemir**: contains the embedding generated by the Jukemir model of the first 30 seconds of the tune it is an array of 4800 dimensions.

- **Embedding\_mule**: contains the embedding generated by the MULE model for the first 30 seconds of the tune; it is an array of 1728 dimensions.

- **Embedding\_mert**: contains the embedding generated by the MERT model for the first 30 seconds of the tune; this is an array of 1024 dimensions.


## Full Dataset Link:

Dataset with all embeddings: https://drive.google.com/file/d/136CftGuB3LuXAKYCy7oCq7m4pNn3BbKI/view?usp=sharing


<!-- CITATION -->
### Citation

[![DOI](https://zenodo.org/badge/700259318.svg)](https://zenodo.org/doi/10.5281/zenodo.10659379)

If you find this repo useful in your research, please consider citing:

```
@article{irish-genres-2024,
  title={Identifying Irish traditional music genres using latent audio representations},
  author={DIEGO M. JIMÉNEZ-BRAVO, ÁLVARO LOZANO MURCIEGO, JUAN JOSÉ
NAVARRO-CÁCERES, MARÍA NAVARRO-CÁCERES, AND TREASA HARKIN,
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```

# About The Project

This repository contains the results of the collaboration between the Irish Traditional Music Archive and researchers from University of Salamanca, as part of the European project "EA-Digifolk".

<br />
<div align="center">
  <a href="https://github.com/elloza/DIGIFOLK-USAL-ITMA">
    <img src="https://usal.es/files/logo_usal.png" alt="Logo" width="250" height="100" style="margin:10px;padding:20px;">
  </a>
  <a href="https://github.com/elloza/DIGIFOLK-USAL-ITMA">
    <img src="https://www.itma.ie/wp-content/themes/ITMA/images/itma-logo.svg" alt="Logo" width="250" height="100" style="margin:10px;padding:20px;">
  </a>
  <a href="https://github.com/elloza/DIGIFOLK-USAL-ITMA">
    <img src="https://cordis.europa.eu/images/logo/logo-ec-es.svg" alt="Logo" width="250" height="100" style="margin:10px;padding:20px;">
  </a>
</div>


# References about utils

We use several python tools to process the data. Here we list the main ones:

* [abc2xml](https://wim.vree.org/svgParse/abc2xml.html)
* [xml2abx](https://wim.vree.org/svgParse/xml2abc.html)

Copyright (C) 2012-2018: W.G. Vree

# Aknowledgements

We would like to thank the European project "EA-Digifolk" for the support and funding of this research.

We would also like to thank the Irish Traditional Music Archive for providing the data and the support for this research.

We would like to thank the Session platform for their work in the promotion of traditional music and for making available the files. All the data from TheSession is available in their website: https://thesession.org/