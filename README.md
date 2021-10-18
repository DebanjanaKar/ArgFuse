# ArgFuse
This repository contains the dataset and source code for the following paper published in ACL 2021 workshop [CASE](https://aclanthology.org/events/acl-2021/#2021-case-1):

#### ["ArgFuse: A Weakly-Supervised Framework for Document-Level Event Argument Aggregation"](https://aclanthology.org/2021.case-1.5/)

ArgFuse presents a neural framework to aggregate event-argument mentions using weakly-supervised approaches in an extractive fashion. To the best of our knowledge,  we are the first to establish baseline results for the task of Event Argument Aggregation in English. We have annotated our own test dataset comprising of 131 document information frames and have released both the code and dataset in this repository. The source code in this repository is based on the hugginface transformer package, BERT and RoBERTa.

ArXiv paper: [https://arxiv.org/abs/2106.10862](https://arxiv.org/abs/2106.10862)

## Contents

+ [Setup](#setup)
+ [Data](#data)
+ [Pipeline](#pipeline)
  - [Relevance Check](#relevance-check)
  - [Redundance Check (active learning)](#redundance-check)
  - [Aggregated data frames](#aggregated-data-frames)
+ [Evaluation](#evaluation)
+ [Citation](#citation)

## Setup

Please use the below command to clone and install the requirements.

```bash
git clone <repo>
cd argfuse/
conda env create -f environment.yml
conda activate aggregate
```

Download the following nltk packages in the virtual environment

```
python
import nltk
nltk.download('punkt')
exit()
python -m spacy download en_core_web_sm
```

## Data

The following table contains the details of all the annotated data published in this repository. All the data mentioned below has been manually annotated for this task.
| File  | Description |
| ------------- | ------------- |
| [english_doc_level.txt](https://github.com/DebanjanaKar/ArgFuse/blob/main/data/english_doc_level.txt)  | annotated test set for event argument aggregation comprising of 131 aggregated document frames |
| [relevance_check.csv](https://github.com/DebanjanaKar/ArgFuse/blob/main/data/relevance_check.csv)  | annotated data for checking if an argument mention is relevant to the topical context of the document |
| [test_rel.csv](https://github.com/DebanjanaKar/ArgFuse/blob/main/data/test_rel.csv) | test set for the relevance classifier |
| [redundance_check.csv](https://github.com/DebanjanaKar/ArgFuse/blob/main/data/redundance_check.csv) | annotated data for checking if a pair of argument mentions contain redundant information |
| [test.csv](https://github.com/DebanjanaKar/ArgFuse/blob/main/data/test.csv) | test set with gold event argument mentions for the weakly-supervised redundancy classifier |
| [pred.csv](https://github.com/DebanjanaKar/ArgFuse/blob/main/data/pred.csv) |  dataset with predicted event argument mentions for the weakly-supervised redundancy classifier |
| [unlabelled.csv](https://github.com/DebanjanaKar/ArgFuse/blob/main/data/unlabelled.csv) | unlabelled dataset to be labelled using active learning for redundancy classifier |
| [en_pred-cas.txt](https://github.com/DebanjanaKar/ArgFuse/blob/main/data/en_pred-cas.txt) | event argument prediction output from [this model](https://aclanthology.org/2020.icon-main.38.pdf) used for aggregation here |

## Pipeline

Move to the `src` directory to run the scripts: `cd src`

To prepare the data, run:
+ `python data_create.py`
+ `python format_data_rel.py` (formats data for relevance check)
+ `python format_data_red.py` (formats data for redundance check)

Due to copyright issue, we could not provide the entire dataset used for [event argument extraction](https://aclanthology.org/2020.icon-main.38.pdf). Hence we have provided a snapshot [here](https://github.com/DebanjanaKar/ArgFuse/blob/main/resources) for better understanding.

#### Relevance Check

To train the relevance classifier, run: `python relevance_check.py --model_type roberta --model_path roberta-base`

#### Redundance Check

To train the redundancy classifier, run the following command iteratively 5 times: 
`python active_learning.py --add_data 0 --num_epochs 1`

In each iteration the most confusing data points are written into a file `sample2label$$.csv` where `$$` is replaced by the number of iteration the script is running for. The number of iteration needs to be mentioned each time while running the script by changing the `add_data` argument. In the ultimate + 1 iteration (5 in our case)
, change the `num_epochs` parameter to 15 to finally train the entire model on the entire dataset (manually annotated + actively annotated).

#### Aggregated data frames

The above two steps are the only training involved in this framework. Here we will mention how we will curate the aggregated document frames.

+ Run `python rank_args.py` to filter relevant argument mentions using the trained relevance classifier and rank them per argument type (using `biased_textrank.py`)
+ Run `python get_doc_frame.py` to generate aggregated document frames with relevant and non-redundant information.

#### Evaluation

Create an evaluation directory: `mkdir results/`

+ Run `python get_output.py` to write the outputs into files.
+ Run `python score.py` on the output files from previous step to print the precision, recall and f1-score values.

## Citation

If you use this code and data in your research, please cite our paper(s):

```
@inproceedings{kar-etal-2021-argfuse,
    title = "{A}rg{F}use: A Weakly-Supervised Framework for Document-Level Event Argument Aggregation",
    author = "Kar, Debanjana  and
      Sarkar, Sudeshna  and
      Goyal, Pawan",
    booktitle = "Proceedings of the 4th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE 2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.case-1.5",
    doi = "10.18653/v1/2021.case-1.5",
    pages = "20--30",
    abstract = "Most of the existing information extraction frameworks (Wadden et al., 2019; Veysehet al., 2020) focus on sentence-level tasks and are hardly able to capture the consolidated information from a given document. In our endeavour to generate precise document-level information frames from lengthy textual records, we introduce the task of Information Aggregation or Argument Aggregation. More specifically, our aim is to filter irrelevant and redundant argument mentions that were extracted at a sentence level and render a document level information frame. Majority of the existing works have been observed to resolve related tasks of document-level event argument extraction (Yang et al., 2018; Zheng et al., 2019) and salient entity identification (Jain et al., 2020) using supervised techniques. To remove dependency from large amounts of labelled data, we explore the task of information aggregation using weakly supervised techniques. In particular, we present an extractive algorithm with multiple sieves which adopts active learning strategies to work efficiently in low-resource settings. For this task, we have annotated our own test dataset comprising of 131 document information frames and have released the code and dataset to further research prospects in this new domain. To the best of our knowledge, we are the first to establish baseline results for this task in English. Our data and code are publicly available at https://github.com/DebanjanaKar/ArgFuse.",
}
```

```
@inproceedings{kar-etal-2020-event,
    title = "Event Argument Extraction using Causal Knowledge Structures",
    author = "Kar, Debanjana  and
      Sarkar, Sudeshna  and
      Goyal, Pawan",
    booktitle = "Proceedings of the 17th International Conference on Natural Language Processing (ICON)",
    month = dec,
    year = "2020",
    address = "Indian Institute of Technology Patna, Patna, India",
    publisher = "NLP Association of India (NLPAI)",
    url = "https://aclanthology.org/2020.icon-main.38",
    pages = "287--296",
    abstract = "Event Argument extraction refers to the task of extracting structured information from unstructured text for a particular event of interest. The existing works exhibit poor capabilities to extract causal event arguments like Reason and After Effects. Futhermore, most of the existing works model this task at a sentence level, restricting the context to a local scope. While it may be effective for short spans of text, for longer bodies of text such as news articles, it has often been observed that the arguments for an event do not necessarily occur in the same sentence as that containing an event trigger. To tackle the issue of argument scattering across sentences, the use of global context becomes imperative in this task. In our work, we propose an external knowledge aided approach to infuse document level event information to aid the extraction of complex event arguments. We develop a causal network for our event-annotated dataset by extracting relevant event causal structures from ConceptNet and phrases from Wikipedia. We use the extracted event causal features in a bi-directional transformer encoder to effectively capture long-range inter-sentence dependencies. We report the effectiveness of our proposed approach through both qualitative and quantitative analysis. In this task, we establish our findings on an event annotated dataset in 5 Indian languages. This dataset adds further complexity to the task by labeling arguments of entity type (like Time, Place) as well as more complex argument types (like Reason, After-Effect). Our approach achieves state-of-the-art performance across all the five languages. Since our work does not rely on any language specific features, it can be easily extended to other languages as well.",
}
```


