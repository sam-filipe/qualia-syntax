# Qualia Syntax

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Supplementary material for:

> **Statistically Modelling Qualia Syntax Through Fourier Space: A Three-role Probabilistic Framework Where Ambiguity Becomes Structure**
>
> Samuel Pereira, Gilberto Bernardes, and José Oliveira Martins

This repository contains the source code, data, and analytical outputs accompanying the article. The study investigates whether composers' deployment of harmonic qualities&mdashincluding diatonicity, octatonicity, and whole-tone quality---exhibits systematic patterns that could constitute a *qualia syntax*. Building upon the discrete Fourier transform (DFT) of pitch-class sets and its geometric interpretation through the Fourier Qualia Space (FQS), it advances three key contributions: the definition of an ambiguity region in the FQS, a computational methodology for extracting qualia progressions from musical works, and a four-tiered nested analysis examining these progressions across multiple analytical scales.

## Repository structure

```
qualia-syntax/
├── src/                        # Analysis pipeline
│   ├── main.py                 # Entry point: orchestrates the full pipeline
│   ├── corpus.py               # YCAC corpus loading and pitch-class vector extraction
│   ├── segmentation.py         # Distance-sensitive and windowed segmentation strategies
│   ├── fourier_qualia_space.py # DFT computation and RadViz (FQS) projection
│   └── analysis.py             # Qualia classification, transition matrices, HCA, Zipf analysis
│
├── tests/                      # Robustness and sensitivity analyses
│   ├── ambiguity_radius.py           # Ambiguity radius sensitivity (rho tests)
│   └── parameter_sensitivity.py  # Segmentation parameter sensitivity
│
├── data/
│   ├── debussy/                # Debussy-specific analytical outputs
│   │   ├── complete_dataframe.csv   # Full processed dataframe for Debussy's catalogue
│   │   ├── conditional_matrix.csv   # Conditional (Markov) transition matrix
│   │   ├── global_matrix.csv        # Global transition percentage matrix
│   │   ├── reflets_segments.csv     # Segment-level analysis of Reflets dans l'eau
│   │   └── regression_data.csv      # Zipf regression data
│   │
│   ├── dendrograms/            # Hierarchical clustering dendrograms (PNG)
│   │   └── {composer}_dendrogram.png  # One per composer (28 composers)
│   │
│   ├── transition_counts/      # Qualia transition count matrices
│   │   └── {composer}_transition_counts.{csv,xlsx}
│   │
│   └── diachronic_zipfian_analysis.csv  # Zipf slopes across composers and periods
│
├── README.md
├── LICENCE                     # MIT licence (source code)
├── LICENCE-DATA                # CC BY 4.0 licence (data files)
├── requirements.txt
└── .gitignore
```

## Getting started

### Prerequisites

- Python 3.11 or later
- The [Yale-Classical Archives Corpus (YCAC)](https://ycac.yale.edu/) installed locally

### Installation

```bash
git clone https://github.com/<your-username>/qualia-syntax.git
cd qualia-syntax
pip install -r requirements.txt
```

### Running the analysis

1. Open `src/main.py` and set the `CORPUS_DIRECTORY` variable to the path of your local YCAC corpus folder.
2. Run the pipeline:

```bash
cd src
python main.py
```

3. Follow the interactive prompts to select the analysis mode (specific composers or entire corpus).

### Running the sensitivity tests

The robustness analyses reported in the article can be reproduced with:

```bash
python tests/qualiaSyntax_radius_tests.py
python tests/qualiaSyntax_parameter_sensitivity.py
```

## Data description

### Debussy analytical data (`data/debussy/`)

| File | Description |
|------|-------------|
| `complete_dataframe.csv` | Full processed dataframe for Debussy's catalogue, including qualia progressions per piece |
| `conditional_matrix.csv` | Conditional (row-normalised) first-order Markov transition probability matrix |
| `global_matrix.csv` | Global transition percentage matrix across Debussy's qualia progressions |
| `reflets_segments.csv` | Segment-by-segment qualia analysis of *Reflets dans l'eau* (Images, L. 110) |
| `regression_data.csv` | Log-rank vs log-frequency regression data for Zipf analysis |

### Dendrograms (`data/dendrograms/`)

Hierarchical clustering dendrograms (Ward's method, Euclidean distance) for 28 composers spanning the Baroque to early 20th-century periods. Each dendrogram visualises the transitional similarity relationships between the seven harmonic qualities.

### Transition counts (`data/transition_counts/`)

First-order qualia transition count matrices for each composer, provided in both CSV and XLSX formats.

### Diachronic Zipfian analysis (`data/diachronic_zipfian_analysis.csv`)

Regression slopes and y-intercepts for log-rank vs log-frequency plots across all analysed composers, ordered chronologically. This file underpins Figures 8 and 9 of the article.

## External resources

This work builds upon and makes use of the following resources:

- **Yale-Classical Archives Corpus (YCAC)**: the primary data source for musical works analysed in this study.
  - Website: <https://ycac.yale.edu/>
  - White, C. W., & Quinn, I. (2016). The Yale-Classical Archives Corpus. *Empirical Musicology Review*, 11(1), 50--58. <https://doi.org/10.18061/emr.v11i1.4958>

- **midiVERTO**: a web application for visualising tonality in real time, whose underlying DFT-based framework informs the Fourier Qualia Space used in this study.
  - Web application: <https://dcmlab.github.io/midiVERTO/#/>
  - Harasim, D., Affatato, G., & Moss, F. C. (2022). midiVERTO: A Web Application to Visualize Tonality in Real Time. In M. Montiel, O. A. Agustin-Aquino, F. Gomez, J. Kastine, E. Lluis-Puebla, & B. Milam (Eds.), *Mathematics and Computation in Music* (Vol. 13267, pp. 363--368). Springer International Publishing. <https://doi.org/10.1007/978-3-031-07015-0_31>

## Licence

Source code is released under the [MIT Licence](LICENCE). Data files are released under the [Creative Commons Attribution 4.0 International Licence](LICENCE-DATA). See the respective licence files for full terms.

## Citation

If you use this code or data in your research, please cite:

```
Pereira, S., Bernardes, G., & Oliveira Martins, J. (2026). Statistically Modelling Qualia Syntax
Through Fourier Space: A Three-role Probabilistic Framework Where Ambiguity Becomes Structure.
```

## Authors

- **Samuel Pereira** -- University of Porto, Faculty of Engineering & INESC TEC
- **Gilberto Bernardes** -- University of Porto, Faculty of Engineering & INESC TEC
- **Jose Oliveira Martins** -- University of Coimbra, Faculty of Arts and Humanities & CEIS20

## Acknowledgements

We are grateful to the University of Porto (UP), the Faculty of Engineering (FEUP), and INESC TEC for the institutional support and resources that made this research possible. Funding was provided by the Foundation for Science and Technology (FCT) through doctoral grant No. 2021.05059.BD (<https://doi.org/10.54499/2021.05059.BD>).
