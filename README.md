# HaMSE Ontology
This repository contains the code developed for the Master Thesis in Digital Humanities and Digital Knwoledge at the University of Bologna, under the supervision of Prof. Aldo Gangemi and the supervision of Prof. Albert Meroño Peñuela.
Afterwards, the work was expanded and revised for the following publication:
```
Andrea Poltronieri, & Aldo Gangemi (2022). The HaMSE Ontology: Using Semantic Technologies to support Music Representation Interoperability and Musicological Analysis. CoRR, abs/2202.05817.
```
More specifically, the repository three main types of resources:
* the code used to extract different features from music content [src folder](#src).
* the [HaMSE ontology](https://raw.githubusercontent.com/andreamust/HaMSE_Ontology/master/schema), which proposes to represent these features
* a small [Knowledge Base](https://raw.githubusercontent.com/andreamust/HaMSE_Ontology/master/data), containing an example of data extracted with the algorithms aforementioned and modelled using the HaMSE Ontology. 

## Data Extraction Algorithms
The data extraction algorithms are divided into five main categories, aiming at extracting five different types of musical features.
The image below summarises these categories and the features belonging to each of them:
<p align="center">
<img src="https://user-images.githubusercontent.com/44606182/162017554-bf518758-3b0c-494b-904b-b4b8e8fa17ff.png" alt="Project Structure" width="500" height="700"/>
</p>


### Audio-to-score Alignment
The Audio-to-score Alignment serves for interconnecting different symbolic music representation systems with the audio representation.
For this task, we used a classic approach based on Dynamic Time Warping (DTW). 
In particular, we have used the DTW algorithm provided by the [Librosa](https://github.com/librosa/librosa).

More specifically, the symbolic notation has been converted to audio signal using the [Midi2audio library](https://github.com/bzamecnik/midi2audio).
The two signals have been then aligned using the DTW algorithm. 
This is an example of the alignment between two tracks:

### Emotional Features

The chosen approach is based on features extraction from raw audio, utilising machine learning techniques. In particular, the approach I followed is based on deep neural networks. The idea behind the extraction of psychoacoustic and emotional features is to train a deep neural network using a dataset labelled with this kind of features, and then make predictions on unseen data. This approach will allow us to understand the emotion aroused by the piece (or a part of a piece) we are analysing.
The dataset chosen for this task is the 4Q audio emotion dataset30 provided by the Centre for Informatics and Systems on the University of Coimbra (Panda, Malheiro and Paiva 2018). The dataset is made up of 900 audio-clips of 30 seconds length, divided into four different groups.
These groups are defined according to the four quadrants of Russell’s model. James A. Russell in 1980 introduced a Circumplex Model of Affect (Russell 1980) that aimed at spatially representing interrelations between affective dimensions.

The best performance in training the model (77.2% in accuracy) were obtained using a Convolutional Recurrent Network, which layers were organised as follows:



### Melodic Features
Extracting melodic features means horizontally investigating the music data, analysing what is functional to the melody.
For these experiments we decided to extract different types of melodic patterns, using a relatively trivial approach based on n-grams. 
To support the scanning of symbolic representations we reused [music21](https://github.com/cuthbertLab/music21) and [The VIS framework](https://github.com/ELVIS-Project/vis-framework). 

The implemented algorithm allows the extraction of different types of melodic patterns, including:
* Interval Patterns
* Rhythmic Patterns
* Identical Patterns
* Longest identical Patterns.

All these types can be extracted either considering rests or not. 

### Harmonic Features

For harmonic features, we implemented an algorithm to extract several harmonic features, including:
* the pitch of the song
* the dissonances present in the song
* the chords that are formed in polyphonic compositions
* the recurrent harmonic sequences, using a mirror approach to that proposed in the previous section. 

### Structural Features

For the structural features of the song, i.e. the structure and sections that make up the song, we reused the [MSAF library](https://pythonhosted.org/msaf/tutorial.html), using it directly on the audio signal. 

## HaMSE Ontology
Starting from the lack of ontologies that deal with musicological research in an extensive way, we propose a novel ontological model.
This new model merges and harmonizes two or more different and non-interoperable music representations (e.g. audio data and scores), either symbolic or non-symbolic. This approach has multiple advantages: by connecting different MSRs, it will be possible to exploit all the characteristics that such representations can provide. Secondly, this novel representation attempts to solve some of the problems that affect contemporary musicology.

The documentation of the ontology, together with other more specific information is hosted on [The HaMSE Project Website](https://andreapoltronieri.org/HaMSE_project/).
The URI of this ontology is the following: `https://github.com/andreamust/HaMSE_Ontology/schema`

## The Knowledge Base
We created a small knowledge base in order to test the ontology. The knowledge base was populated with data from a Bach’s chorale, namely the sixth movement of the church cantata Wie schön leuchtet der Morgenstern. The piece is catalogued under the index BWV1. The sixth and last movement is the chorale entitled *"Wie bin ich doch so herzlich froh"*.


## Citations

```
@article{poltronieri2022HaMSEOntology,
          author    = {Andrea Poltronieri and
                       Aldo Gangemi},
          title     = {The HaMSE Ontology: Using Semantic Technologies to support Music Representation
                       Interoperability and Musicological Analysis},
          journal   = {CoRR},
          volume    = {abs/2202.05817},
          year      = {2022},
          url       = {https://arxiv.org/abs/2202.05817},
          eprinttype = {arXiv},
          eprint    = {2202.05817},
          timestamp = {Fri, 18 Feb 2022 12:23:53 +0100},
          biburl    = {https://dblp.org/rec/journals/corr/abs-2202-05817.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
}

```

A more advanced version of this work can be found in:

```
@inproceedings{poltronieri2021musicnoteontology,
          title={The Music Note Ontology},
          author={Poltronieri, Andrea and Gangemi, Aldo},
          booktitle={Proceedings of the 12th Workshop on Ontology Design and Patterns (WOP 2021), Online, October 24, 2021.},
          journal={CEUR-WS},
          editor={Hammar, Karl and Shimizu, Cogan and Küçük McGinty, Hande and Asprino, Luigi and Carriero, Valentina Anita},
          year={2021},
          month={11}
}
```

## License
MIT License

Copyright (c) 2021 Andrea Poltronieri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
