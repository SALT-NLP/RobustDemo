# Robustness of Demonstration-based Learning Under Limited Data Scenario

This repo contains codes for the following paper: 

*Hongxin Zhang, Yanzhe Zhang, Ruiyi Zhang and Diyi Yang*: 	Robustness of Demonstration-based Learning Under Limited Data Scenario, EMNLP 2022

```
@misc{zhang2022robustness,
    title={Robustness of Demonstration-based Learning Under Limited Data Scenario},
    author={Hongxin Zhang and Yanzhe Zhang and Ruiyi Zhang and Diyi Yang},
    year={2022},
    eprint={2210.10693},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

If you would like to refer to it, please cite the paper mentioned above. 

## Dependency

To run our code, please install all the dependency packages and activate the environment with the following command:

```
conda env create -f env.yaml
conda activate RobustDemo
```

## Prepare the data

1. Download the [CoNLL03 dataset](https://www.clips.uantwerpen.be/conll2003/ner/), [OntoNotes 5.0 dataset](https://catalog.ldc.upenn.edu/LDC2013T19), [NRB(WTS) dataset](https://drive.google.com/file/d/1lOU6dfTyQ-R1Ie8UFDXwNQPNxeBTNIne/view) for NER task and [CoNLL00 dataset](https://www.clips.uantwerpen.be/conll2000/chunking/) for Chunking task and prepare them in the data folder properly as follows:
```
|__ data/
        |__ chk/
                |__ conll2000/
        |__ ner/
                |__ conll2003/
                |__ NRB_WTS/
                |__ ontonotes/
```
2. Use the following command (in the root directory) to generate the few-shot data we need:
```bash
./tools/generate_few_shot.sh
```

See tools/sample_greedy.py for more options.

## Demonstration mode

| Mode             | Template                                   | Description                              |
|------------------|--------------------------------------------|------------------------------------------|
| `standard`       | `[SEP] {context} {entity} is {tag}.`       | Standard demonstration                   |
| `standard_wrong` | `[SEP] {context} {entity} is {wrong_tag}.` | Standard demonstration with Wrong labels |
| `standard_no_l`  | `[SEP] {context}`                          | Standard demonstration with No label     |
| `random_totally` | `[SEP] {totally_randomized_context}`       | Totally Random demonstration             |
| `random_support` | `[SEP] {relevant_randomized_context}`      | Support set sampled Random demonstration |

<hr/>

## Usage

### Generate Demonstrations
To generate all 5 modes of demonstration mentioned above for model BERT and dataset conll2003, use the following command:

```bash
python3 tools/make_demonstration.py --model bert --task ner --dataset conll2003 --k 5shots --mode standard standard_wrong standard_no_l random_totally random_support
```

### Experiments with multiple runs

To carry out experiments with 5 different data splits and 3 different random seeds on dataset conll2003 with model bert-base-cased, run the following command:

```bash
./exp.sh
```

Results will be in the output_dir specified within the script, then use the following command to aggregate the results and have the visualizations:

```bash
python3 tools/statistics.py --task ner --dataset conll2003 --k 5shots --mode no standard standard_wrong standard_no_l random_totally random_support
```

See tools/sample_greedy.py for more options.
