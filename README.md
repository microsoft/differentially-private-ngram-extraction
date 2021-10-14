# Differentially Private n-gram Extraction
This is a repository for implementing Differentially Private N-grams Extraction (DPNE) paper ([preprint version](https://arxiv.org/abs/2108.02831)), to appear in NeurIPS 2021.

# Directory structure

The code repository structure is as follows:
- dpne: has python codes to run each step for extracting DPNE n-grams with PySpark code.
  - dpne_utils.py: has generic util functions used across differnt scripts
  - extract_dpne.py: implements main algorithm of DPNE
  - gaussian_process.py: calcualtes gaussian noise to be added for each n-gram
  - k_anon_coverage.py: generates k-anonymized n-grams, calculates the coverage of DPNE n-grams agains k-anonmized n-grams
  - split_ngrams.py: split into subfolder for each size of the tokenized n-grams (input preparation)
  - tokenize_text.py: tokenizs text with nltk tokenizer
- scripts: has scripts to run the code
  - convert_msnbc.py: converts MSNBC data
  - convert_reddit.py: converts Reddit data 

# Prerequisites

The code requires following libraries installed:
- python >= 3.6
- nltk
- numpy
- PySpark == 2.3
- shrike

# How to run

## Prepare data
First, download the data from below:
- MSNBC: https://archive.ics.uci.edu/ml/datasets/msnbc.com+anonymous+web+data
- Reddit: https://github.com/webis-de/webis-tldr-17-corpus, 
downloadable from https://zenodo.org/record/1043504/files/corpus-webis-tldr-17.zip

Then run the convert scripts from DPNE home directory,
```
python scripts/convert_msnbc.py --input_path [Input file path which has the downloaded file] --output_path [output directory, like /output]
python scripts/convert_reddit.py --input_path [Input file path which has the downloaded file] --output_path [output directory, like /output]
```

## Run DPNE

1. Archive the dpne directory to dpne.zip, this is needed for PySpark to use the package of the whole python scripts
2. Use run.cmd, you will need to modify the first line of DATA_HOME where your converted data exists. Simply you can run it below from DPNE home directory,
```
.\scripts\run.cmd
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
