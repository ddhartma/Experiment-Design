[image1]: assets/ab_test.png "image1"
[image2]: assets/ab_and_or.png "image2"
[image3]: assets/Simple_Random_Sampling.png "image3"
[image4]: assets/Stratified_Random_Sampling.png "image4"
[image5]: assets/goal_metrics.png "image5"
[image6]: assets/funnel.png "image6"
[image7]: assets/unit_of_diversion.png "image7"
[image8]: assets/metrics.png "image8"
[image9]: assets/controlling_vars.png "image9"
[image10]: assets/construct_val.png "image10"
[image11]: assets/internal_val.png "image11"
[image12]: assets/external_val.png "image12"
[image13]: assets/sampling_bias.png "image13"
[image14]: assets/novelty_bias.png "image14"
[image15]: assets/order_bias.png "image15"
[image16]: assets/norm.png "image16"

# Experimental Design
Within the experimental design portion of this course, there are three lessons:

## Outline
- [Concepts of Experiment Design](#Concepts_of_Experiment_Design)
  - [What is an experiment](#What_is_an_experiment)
  - [What are the types of experiments?](#Types)
  - [Types of Sampling](#Types_of_Sampling)
  - [Measuring Outcomes](#Measuring_Outcomes)
  - [Creating metrics](#Creating_Metrics)
  - [Controlling variables](#Controlling_variables)
  - [Checking validity](#Checking_validity)
  - [Checking bias](#Checking_bias)
  - [Ethics in Experimentation](#Ethics)
  - [A SMART Mnemonic for Experiment Design](#SMART)

- [Statistical Considerations in Testing](#Statistical_Considerations_in_Testing)
  - Statistical techniques and considerations used when evaluating the data collected during an experiment.
  - Applying inferential statistics in different ways.

- A/B Testing Case Study
  - Analyze data related to a change on a web page designed to increase purchasers of software.

- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

# Concepts of Experiment Design
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

- For example, you might include a survey to random users to have them rate their website experience for an additional feature on a 1-10 scale. If the addition is helpful, then we should expect the average rating to be higher for those users who are given the addition, versus those who are not. The rating scale acts as a concrete way of measuring user satisfaction. These objective features by which you evaluate performance are known as ***evaluation metrics***.
- ***Evaluation metrics*** = Features that provide an objective measure of the success of an experimental manipulation

  ![image5]

- As a rule of thumb, it's a good idea to consider the ***goals*** of a study ***separate*** from the ***evaluation metrics***. 
- It's the ***implications of the metric*** relative to the ***goal*** that matters.

- ***Alternate Terminology for evaluation metrics***: 
  - construct (social sciences)
  - key results (KRs) or key performance indicators (KPIs) as ways of measuring progress against quarterly or annual objectives

- Further study: [When You Experiment with The Wrong Metrics…](https://hackernoon.com/when-you-experiment-with-the-wrong-metrics-85c51cc594ee)


## Creating Metrics <a name="Creating_Metrics"></a>

There are further terms commonly used for designing experiments:

- ***Funnel***: A funnel is the flow of steps you expect a user of your product to take
Example: Online store
  - Visit the site homepage
  - Search for a desired product or click on a product category
  - Click on a product image
  - Add the product to the cart
  - Check out and finalize purchase

  ![image6]

  One property to note about user funnels is that typically there will be some dropoff in the users that move from step to step. This is much like how an actual funnel narrows from a large opening to a small exit. Outside of an experiment, funnels can be used to analyze user flows. Observations from these flows can then be used to motivate experiments to try and improve the dropoff rates.

  It's also worth noting that the flow through a funnel might be idealized compared to actual user practice. In the above example, users might perform multiple searches in a single session, or want to purchase multiple things. A user might access the site through a specific link, subverting the top part of the funnel. Refining the funnel and being specific about the kinds of events that are expected can help you create a consistent, reliable design and analysis.

- ***Unit of Diversion***: Way to assign users to either a control group or experimental group

  - ***Event-based diversion*** (e.g. pageview): Each time a user loads up the page of interest, the experimental condition is randomly rolled. Since this ignores previous visits, this can create an inconsistent experience, if the condition causes a user-visible change.
  - ***Cookie-based diversion***: A cookie is stored on the user's device, which determines their experimental condition as long as the cookie remains on the device. Cookies don't require a user to have an account or be logged in, but can be subverted through anonymous browsing or a user just clearing out cookies.
  - ***Account-based diversion*** (e.g. User ID): User IDs are randomly divided into conditions. Account-based diversions are reliable, but requires users to have accounts and be logged in. This means that our pool of data might be limited in scope, and you'll need to consider the risks of using personally-identifiable information.


  ![image7]

    In tis example Unit of Diversion = cookies

- ***Evaluation Metrics***: 
    - metrics on which we will compare the two groups)
    - these are the values that will tell us if our experiment is a success or not
    - we are not limited to just one metric, one can track multiple metrics

- Define also ***Invariant Metrics***:
    - metrics that do not change between groups
    - these metrics are useful to verify that control and experimental groups are comparable to one another except for for the experimental condition
    - e.g. number of cookies (each website version should get same number of visitors due to randomization)

    ![image8]


## Controlling variables <a name="Controlling_variables"></a>
[Correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation): Learn about potential problems that can weaken a study`s usefulness
- A ***confounding variable*** is one that is not part of the variables of interest but has an interaction with them

- In other words: The correlation observed between two variables might be due to changes in a third variable, rather than one causing the other.

    ![image9]
    
    In summer time: Selling icecream shows a similar increase like comitting crimes. There is now causality between these both variables, but on a third one temperature. A higher tmeperature favours selling icecream but leads also to more unrelaxed people and therefore crimes.

## Checking validity <a name="Checking_validity"></a>

When you explain the outcome to others, outcomes must be valid. 
Validity concerns how well conclusions can be supported, or the degree to which your experiment actually accomplishes the conclusions you state. There are three major conceptual dimensions upon which validity can be assessed:
- Construct Validity
- Internal Validity
- External Validity

### Construct Validity
- how well one's goals are aligned to the evaluation metrics

    ![image10]

### Internal Validity
- Internal validity refers to the degree to which a causal relationship can be derived from an experiment's results

- Here an example for poor internal validity

    ![image11]

    poor internal validity would be to say: Ice cream sales cause an increase in crimes

- Be aware, not to take correlations as causality without taking other variables into account


### External Validity 
- External validity is concerned with the ability of an experimental outcome to be generalized to a broader population. This is most relevant with experiments that involve sampling: how representative is the sample to the whole? For studies at academic institutions, a frequent question is if data collected using only college students can be generalized to other age or socioeconomic groups.

    ![image12]

    How represenative is a test score at one university comparable to other universities?


## Checking bias <a name="Checking_bias"></a>
Biases in experiments are systematic effects that interfere with the interpretation of experimental results, mostly in terms of internal validity. Just as humans can have a lot of different biases, there are numerous ways in which an experiment can become unbalanced.

Bias in an experiment is always bad. It leads to systematic errors that affect the interpretation of your results.


### Sampling bias
- Sampling biases are those that cause our observations to not be representative of the population.
- e.g. if assignment to experimental groups is done in an arbitrary fashion (as opposed to random assignment or matched groups), we risk our outcomes being based less on the experimental manipulation and more on the composition of the underlying groups.

    ![image13]

    Here we introduce a bias if assign people to control and experiment group not by random but via time periods. 

### Novelty bias
A novelty effect is one that causes observers to change their behavior simply because they're seeing something new. 
- We might not be able to gauge the true effect of a manipulation until after the novelty wears off and population metrics return to a level that actually reflects the changes made. This will be important for cases where we want to track changes over time, such as trying to get users to re-visit a webpage or use an app more frequently. Novelty is probably not a concern (or perhaps what we hope for) when it comes to manipulations that are expected to only have a one-shot effect.

    ![image14]


### Order biases
There are a couple of biases to be aware of when running a within-subjects experiment. 
- Recall that in a within-subjects design, each participant performs a task or makes a rating in multiple experimental conditions, rather than just one. 
- The order in which conditions are completed could have an effect on participant responses. 
- A ***primacy effect*** is one that affects early conditions,  better in mind because it was first seen
- A ***recency effect*** is one that affects later conditions, perhaps causing bias due to being fresher in minds or task fatigue.

    ![image15]

    - "How does this compare to A" --> Primacy Bias
    - "I don't recall A or B well" --> Recency Bias

    An easy way of getting around order biases is to simply randomize the order of conditions. If we have three conditions, then each of the six ways of completing the task (ABC, ACB, BAC, BCA, CAB, CBA) should be equally likely. While there still might end up being order effects like carry-over effects, where a particular condition continues to have an effect on future conditions, this will be much easier to detect than if every participant completed the task in the exact same order of conditions.


## Ethics in Experimentation <a name="Ethics"></a>
Before you run an experiment, it's important to consider the ethical treatments to which you subject your participants. Through the mid-20th century, exposure of questionable and clearly unethical research in the social and medical sciences spurred the creation of guidelines for ethical treatment of human subjects in studies and experiments. While different fields have developed different standards, they still have a number of major points in common:

- ***Minimize participant risk***: Experimenters are obligated to construct experiments that minimize the risks to participants in the study. Risk of harm isn't just in the physical sense, but also the mental sense. If an experimental condition has potential to negatively affect a participant's emotions or mentality, then it's worth thinking about if the risks are really necessary to perform the desired investigation.

- ***Have clear benefits for risks taken***: In some cases, risks may be unavoidable, and so they must be weighed against the benefits that may come from performing the study. When expectations for the study are not clearly defined, this throws into question the purpose of exposing subjects to risk. However, if the benefits are shown to be worth the risks, then it is still possible for the study to be run. This often comes up in medicine, where test treatments should show worthy potential against alternative approaches.

- ***Provide informed consent***: Building up somewhat from the previous two points, subjects should be informed of and agree to the risks and benefits of participation before they join the study or experiment. This is also an opportunity for a participant to opt out of participation. However, there are some cases where deception is necessary. This might be to avoid biasing the participant's behavior by seeding their expectations, or if there is a dummy task surrounding the actual test to be performed. In cases like this, it's important to include a debriefing after the subject's participation so that they don't come away from the task feeling mislead.

- ***Handle sensitive data appropriately***: If you're dealing with identifiable information in your study, make sure that you take appropriate steps to protect their anonymity from others. Sensitive information includes things like names, addresses, pictures, timestamps, and other links from personal identifiers to account information and history. Collected data should be anonymized as much as possible; surveys and census results are often also aggregated to avoid tracing outcomes back to any one person.

## A SMART Mnemonic for Experiment Design <a name="SMART"></a>

There's a mnemonic called SMART for teams to plan out projects that also happens to apply pretty well for creating experiments. The letters of SMART stand for:

- ***Specific***: Make sure the goals of your experiment are specific.
- ***Measurable***: Outcomes must be measurable using objective metrics
- ***Achievable***: The steps taken for the experiment and the goals must be realistic.
- ***Relevant***: The experiment needs to have purpose behind it.
- ***Timely***: Results must be obtainable in a reasonable time frame.




# Statistical Considerations in Testing <a name="Statistical_Considerations_in_Testing"></a>
Statsistics is not only needed to analyse the data. It is also needed to set up an experiment.

## [Inference statistics in Python](https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce)
1. ***What is hypothesis testing?***

    Hypothesis testing is a statistical method that is used in making statistical decisions using experimental data. Hypothesis Testing is basically an assumption that we make about the population parameter. We need some mathematical conclusion what ever we are assuming is true.

    Ex : you say avg student in class is 40 or a boy is taller than girls.

2. ***Why do we use it?***

    Hypothesis testing is an essential procedure in statistics. A hypothesis test evaluates two mutually exclusive statements about a population to determine which statement is best supported by the sample data. When we say that a finding is statistically significant, it’s thanks to a hypothesis test.

3. ***What are the basics of hypothesis?***

    The basic of hypothesis is [normalisation](https://en.wikipedia.org/wiki/Normalization_(statistics)) and [standard normalisation](https://stats.stackexchange.com/questions/10289/whats-the-difference-between-normalization-and-standardization). All hypothesis is based on these 2 terms. 

    ![image16]

    - a standardized normal distribution has a ***mean = 0*** and ***variance = 1***. 
    - This concept of z-score comes in picture when we use standardised normal data.

- ***Normal Distribution***
    1. mean = median = mode
    2. Transformation: 

    $$\int_\Omega \nabla u \cdot \nabla v~dx = \int_\Omega fv~dx$$





## How can statistics be used to set up an experiment? 





























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
## Links
* [Correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation)



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
