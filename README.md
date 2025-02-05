# Differentially Private n-gram Extraction
This is a repository for implementing Differentially Private N-grams Extraction (DPNE) paper ([preprint version](https://arxiv.org/abs/2108.02831)), to appear in NeurIPS 2021.

## Directory structure

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
- DPNE Experiments.ipynb: Jupyter notebook to run on pyspark container for experimentation, which also runs the scripts in `run.cmd` (see below for instructions on how to run this)

## Prerequisites

The code requires following libraries installed:
- python >= 3.6
- nltk
- numpy
- PySpark == 2.3
- shrike

## Preparing container to run experiments
Running this code on a container makes getting started fairly easy and reliably. Follow these steps to get this running on a local container:
- Make sure you have [docker installed](https://docs.docker.com/engine/install/) and running
- Run `docker pull jupyter/pyspark-notebook` to install the pyspark-jupyter container
- Run the container mapping port 8888 locally so you can run the notebook on your machine, using `docker run -p 8888:8888 --name jupyter-pyspark jupyter/pyspark-notebook` - from the logs that open up, paste the command into your browser to run the notebook - something like `http://127.0.0.1:8888/lab?token=<TOKEN>`
- Bash into the container by running `docker exec -it jupyter-pyspark bash`
- Run `git clone https://github.com/microsoft/differentially-private-ngram-extraction.git` to pull this repo into the container
- Install the required libraries as mentioned above:
  ```
  pip install nltk
  pip install pyspark
  pip install shrike==1.31.18
  ```
  Additionally, run a python shell and run the following commands:
  ```
  import nltk
  nltk.download('punkt_tab')
  ```
Now at this point, you can replicate results from the paper or run DP N-grams extraction on your own dataset. See instructions for each case below:

### Replicate results from the paper
There are two data sources cited:
- MSNBC: https://archive.ics.uci.edu/ml/datasets/msnbc.com+anonymous+web+data
- Reddit: https://github.com/webis-de/webis-tldr-17-corpus, 
downloadable from https://zenodo.org/record/1043504/files/corpus-webis-tldr-17.zip
To prepare data from these in the right format, the following scripts from DPNE home directory are used. 
```
python scripts/convert_msnbc.py --input_path [Input file path which has the downloaded file] --output_path [output directory, like /output]
python scripts/convert_reddit.py --input_path [Input file path which has the downloaded file] --output_path [output directory, like /output]
```
This is simplified within the attached notebook, where you can simply follow these steps to run this:
 - With the container running, navigate to the [notebook](http://127.0.0.1:8888/lab/tree/differentially-private-ngram-extraction/DPNE%20Experiments.ipynb) and run the code starting from the first cell which downloads and prepares data (for the reddit case).
 - You may also change the default values of the variables `DP_EPSILON` and `NGRAM_SIZE_LIMIT` based on your needs. Run the commands in the cells which should eventually provide you with the extracted DP n-grams in the `DPNGRAMS` dictionary - `DPNGRAMS["1gram"]` will be a pandas dataframe with the extracted DP 1-grams and so on.
 - Follow the steps in the subsequent cells, which break up the tokenization, splitting of n-grams and then DP n-grams extraction into separate spark sessions, and cache the results locally.
 - Once these scripts have successfully run, the 3rd cell allows reads them into a dictionary of pandas dataframes, from where you may access the extracted DP n-grams.


### Run on your own dataset
- Copy over into the differentially-private-ngram-extraction folder a dataset as a newline delimited JSON file with keys "author" and "content" representing the distinct author name/id, and their content you want to extract DP n-grams from, respectively. On another terminal you can use the command `docker cp /path/to/file.json jupyter-pyspark:/home/jovyan/differentially-private-ngram-extraction/`
- Now you can simply navigate to the [notebook](http://127.0.0.1:8888/lab/tree/differentially-private-ngram-extraction/DPNE%20Experiments.ipynb) and run the code, changing `SOURCE_DATASET` to the name of the JSON file you just copied. If you are using something other than JSON, please change `FILE_EXTENSION` accordingly. You may also change the default values of the variables `DP_EPSILON` and `NGRAM_SIZE_LIMIT` based on your needs. Run the commands in the cells which should eventually provide you with the extracted DP n-grams in the `DPNGRAMS` dictionary - `DPNGRAMS["1gram"]` will be a pandas dataframe with the extracted DP 1-grams and so on.
- Follow the steps in the subsequent cells, which break up the tokenization, splitting of n-grams and then DP n-grams extraction into separate spark sessions, and cache the results locally.
- Once these scripts have successfully run, the 3rd cell allows reads them into a dictionary of pandas dataframes, from where you may access the extracted DP n-grams.

### Run DPNE without the container
If you choose to run this within the shell or with local modifications without using the container method described above, simply follow these steps
1. If you made changes to any file in the dpne/ folder, re-archive the dpne directory to dpne.zip, this is needed for PySpark to use the package of the whole python scripts.
2. Assuming you are on a windows machine, use run.cmd, you will need to modify the first line of DATA_HOME where your converted data exists. Simply you can run it below from DPNE home directory,
```
.\scripts\run.cmd
```
If you are on a Linux based environment, see the corresponding shell scrips in the notebook.

## DPNE + Rephrase experiment with Wildchat Dataset
a. To run the DPNE+Rephrase experiment first install a python version 3.8+ and then install the below libraries as requirements:

```
1. azure-identity==1.16.0
2. dask==2023.5.0
3. datasets==2.19.2
4. huggingface-hub==0.28.1
5. openai==1.33.0
6. sentence-transformers==3.0.0
7. pandas==2.0.3
8. numpy==1.23.5
```

b. Use the .\scripts\run_wildchat_rephrase_exp.cmd script to run the experiment.
## References
[1] Kunho Kim, Sivakanth Gopi, Janardhan Kulkarni, Sergey Yekhanin. Differentially Private n-gram Extraction. In Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS), 2021.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
