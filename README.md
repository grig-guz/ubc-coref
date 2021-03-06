# Coreference Resolution

This repository contains PyTorch reimplementation of the EMNLP paper ["BERT for Coreference Resolution: Baselines and Analysis"](https://arxiv.org/abs/1908.09091) by Joshi et al., 2019. The code is built upon [this repository](https://github.com/shayneobrien/coreference-resolution/) and involves substantial modifications and bug corrections, and rests upon [Span-BERT](https://arxiv.org/abs/1907.10529) as the document encoder.

# Data
The source code assumes access to the English train, test, and development data of OntoNotes Release 5.0. This data should be located in a folder called 'data' inside the main directory. The data consists of 2,802 training documents, 343 development documents, and 348 testing documents. The average length of all documents is 454 words with a maximum length of 4,009 words. The number of mentions and coreferences in each document varies drastically, but is generally correlated with document length.

Since the data require a license from the Linguistic Data Consortium to use, they are thus not supplied here. Information on how to download it can be found [here](https://catalog.ldc.upenn.edu/LDC2013T19). Run the the following script to preprocess it (Note: requires Python 2):
```
bash preprocess_conll.sh PATH_TO_ONTONOTES_5.0 data/ english 
```
Then run the following to preprocess the data into the format used here and save it:
```
python prepare_data.py 
```
# Training
First, install the requirements as specified in ```requirements.txt```
You can start training as follows:
```
python main.py --train 
```
You can add ```--pretrained_coref_path PATH ``` with the path to the model save if the training was interrupted. The pretrained model can be downloaded [here](https://drive.google.com/file/d/1MDvxH3T1R_CC6NKc4tmX8eM_xD-hlBMS/view?usp=sharing).

# Testing

```
python main.py --test --pretrained_coref_path PATH
```

