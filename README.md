[image1]: assets/ab_test.png "image1"
[image2]: assets/ab_and_or.png "image2"
[image3]: assets/Simple_Random_Sampling.png "image3"
[image4]: assets/Stratified_Random_Sampling.png "image4"
[image5]: assets/goal_metrics.png "image5"

# Experimental Design
Within the experimental design portion of this course, there are three lessons:

## Outline
- [Concepts of Experiment Design](#Concepts_of_Experiment_Design)
  - [What is an experiment](#What_is_an_experiment)
  - [What are the types of experiments?](#Types)
  - [Types of Sampling](#Types_of_Sampling)
  - [Measuring Outcomes](#Measuring_Outcomes)
  - [Creating metrics](#Creating_Metrics)
  - What are common pitfalls in experiment design?

II. Statistical Considerations in Testing
  - Statistical techniques and considerations used when evaluating the data collected during an experiment.
  - Applying inferential statistics in different ways.

III. A/B Testing Case Study
  - Analyze data related to a change on a web page designed to increase purchasers of software.

  - [Setup Instructions](#Setup_Instructions)
  - [Acknowledgments](#Acknowledgments)
  - [Further Links](#Further_Links)


## What is an experiment? <a name="What_is_an_experiment"></a>
Key features of an experiment:
1.	***Comparison***: Comparison between groups
2.	***Random assignement to groups***: Control other variables via randomization (by assigning visitors to one or the other webpage other variables like age, gender etc. should be equally distributed). The only practical feature between the groups should be the feature we care about.

Types of study depend on the amount of control over the variables in play.
- ***Experiment***  ---  High level control of features (Web page without overlay vs. web page with overlay)
- ***Observational study*** --- no control (e.g. medical studies, effects of smoking on health)
- ***Quasi-experiment*** --- some control (other features may have an effect on the target)

## What are the types of experiments? <a name="Types"></a>
### A/B Test
- Typical situation of an A/B Test

  ![image1]

  - ***Control group***: A-group which will get old data
  - ***Experimental group***: B-group which will get new data
  - More than two groups are possible ABC (A=control, B/C=experimental)

- Between-subject vs within-subject design

  ![image2]


## Types of Sampling <a name="Types_of_Sampling"></a>
While web and other online experiments have an easy time collecting data, collecting data from traditional methods involving real populations is a much more difficult proposition. The goal of sampling is to use a subset of the whole population to make inferences about the full population, so that we didn't need to record data from everyone
- ***Simple Random Sampling***

  ![image3]

  Randomly make draws from the population until the desired sample size

- ***Stratified Random Sampling***

  ![image4]

  Subgroups are for example needed due to different population (sample) densities
- ***Non-Proportional Sampling***

- ***Non-Probabilistic Sampling***

## Measuring Outcomes <a name="Measuring_Outcomes"></a>
The ***goals*** of your study may not be the same as the way you evaluate the study's ***success***, e.g. because the goal can't be measured directly.

- For example, you might include a survey to random users to have them rate their website experience on a 1-10 scale. If the addition is helpful, then we should expect the average rating to be higher for those users who are given the addition, versus those who are not. The rating scale acts as a concrete way of measuring user satisfaction. These objective features by which you evaluate performance are known as ***evaluation metrics***.
- ***Evaluation metrics*** = Features that provide an objective measure of the success of an experimental manipulation

  ![image5]

- As a rule of thumb, it's a good idea to consider the ***goals*** of a study ***separate*** from the ***evaluation metrics***. 
- It's the ***implications of the metric*** relative to the ***goal*** that matters.

- ***Alternate Terminology for evaluation metrics***: 
  - construct (social sciences)
  - key results (KRs) or key performance indicators (KPIs) as ways of measuring progress against quarterly or annual objectives

- Further study: [When You Experiment with The Wrong Metrics…](https://hackernoon.com/when-you-experiment-with-the-wrong-metrics-85c51cc594ee)


## Creating Metrics <a name="Creating_Metrics"></a>












## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit
- If you need a Command Line Interface (CLI) under Windows you could use [git](https://git-scm.com/). Under Mac OS use the pre-installed Terminal.

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Disaster-Response-Pipeline-Project.git
```

- Change Directory
```
$ cd Disaster-Response-Pipeline-Project
```

- Create a new Python environment, e.g. ds_ndp. Inside Git Bash (Terminal) write:
```
$ conda create --name ds_ndp
```

- Activate the installed environment via
```
$ conda activate ds_ndp
```

- Install the following packages (via pip or conda)
```
numpy = 1.17.4
pandas = 0.24.2
scikit-learn = 0.20
pipelinehelper = 0.7.8
```
Example via pip:
```
pip install numpy
pip install pandas
pip install scikit-learn==0.20
pip install pipelinehelper
```
scikit-learn==0.20 is needed for sklearns dictionary output (output_dict=True) for the classification_report. Earlier versions do not support this.


Link1 to [pipelinehelper](https://github.com/bmurauer/pipelinehelper)

Link2 to [pipelinehelper](https://stackoverflow.com/questions/23045318/scikit-grid-search-over-multiple-classifiers)

- Check the environment installation via
```
$ conda env list
```

### Switch the pipelines
- Active pipeline at the moment: pipeline_1 (Fast training/testing) pipeline
- More sophisticated pipelines start with pipeline_2.
- Model training has been done with ```pipeline_2```.
- In order to switch between pipelines or to add more pipelines open train.classifier.py. Adjust the pipelines in `def main()` which should be tested (only one or more are possible) via the list ```pipeline_names```.

```
def main():
   ...

    if len(sys.argv) == 3:
        ...

        # start pipelining, build the model
        pipeline_names = ['pipeline_1', 'pipeline_2']
```


### Run the web App

1. Run the following commands in the project's root directory to set up your database and model.

    To run ETL pipeline that cleans data and stores in database


        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

    To run ML pipeline that trains classifier and saves

        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


2. Run the following command in the app's directory to run your web app

        python run.py




3. Go to http://0.0.0.0:3001/

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>
Working with Text data
* [Working With Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
* [Natural Language Precessing Book](http://www.nltk.org/book/)
* [Parts of Speech Tagging](https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b)

CountVectorizer and TfidfTransformer
* [sklearn CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [10+ Examples for Using CountVectorizer](https://kavita-ganesan.com/how-to-use-countvectorizer/#.X--j6OAxmFo)
* [sklearn TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
* [TF IDF | TFIDF Python Example](https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76)
* [How to Use Tfidftransformer & Tfidfvectorizer?](https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.X-9OH-AxmFp)

Inherit from Transformer (CountVectorizer)
* [How to inherit from CountVectorizer I](https://stackoverflow.com/questions/51430484/how-to-subclass-a-vectorizer-in-scikit-learn-without-repeating-all-parameters-in)
* [How to inherit from CountVectorizer II](https://sirinnes.wordpress.com/2015/01/22/custom-vectorizer-for-scikit-learn/)
* [Hacking Scikit-Learn’s Vectorizers](https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af)

Pipeline
* [Text Feature Extraction With Scikit-Learn Pipeline](https://towardsdatascience.com/the-triune-pipeline-for-three-major-transformers-in-nlp-18c14e20530)
* [sklearn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
* [Hyperparameter tuning in pipelines with GridSearchCV](https://ryan-cranfill.github.io/sentiment-pipeline-sklearn-5/)

Pipelinehelper
* [pipelinehelper I](https://github.com/bmurauer/pipelinehelper)
* [pipelinehelper II](https://stackoverflow.com/questions/23045318/scikit-grid-search-over-multiple-classifiers)

GridSearchCV
* [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)

Classification
* [Choosing the Best Algorithm for your Classification Model](https://medium.com/datadriveninvestor/choosing-the-best-algorithm-for-your-classification-model-7c632c78f38f)

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Data Visualization
* [10 Python Data Visualization Libraries for Any Field | Mode](https://mode.com/blog/python-data-visualization-libraries/)
* [5 Quick and Easy Data Visualizations in Python with Code](https://towardsdatascience.com/5-quick-and-easy-data-visualizations-in-python-with-code-a2284bae952f)
* [The Best Python Data Visualization Libraries](https://www.fusioncharts.com/blog/best-python-data-visualization-libraries/)
