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

[image16]: assets/A_B_Test.png "image16"
[image17]: assets/metric_hypo.png "image17"
[image18]: assets/experiment_1.png "image18"
[image19]: assets/experiment_2.png "image19"
[image20]: assets/stratified_rnd_sample.png "image20"
[image21]: assets/practical_significance.png "image21"
[image22]: assets/practical_significance_2.png "image22"
[image23]: assets/stat_power.png "image23"
[image24]: assets/dummy_test.png "image24"
[image25]: assets/multiple_metrics.png "image25"
[image26]: assets/multiple_metrics_2.png "image26"
[image27]: assets/early_stopping.png "image27"
[image28]: assets/metrics_ab.png "image28"


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
    - [Binomial Distribution](#Binomial_Distribution)
    - [Computing p-value via stats.norm](#Computing_p_value)
    - [Statistical significance?](#Statistical_significance)
    - [Checking Invariant metric](#Checking_Invariant_metric)
        - [Analytic Approach](#Analytic_Approach)
        - [Simulation Approach](#Simulation_Approach)
    - [Checking the Evaluation Metric](#Checking_the_Evaluation_Metric)
        - [Analytic Approach](#Analytic_Approach_2)
        - [Simulation Approach](#Simulation_Approach_2)
    - [Practical statistical significance?](#Practical_statistical_significance?)
    - [Experiment Size](#Experiment_Size)
        - [Method 1: Trial and Error](#Experiment_Size_trial_error)
        - [Method 2: Analytic Solution](#Experiment_Size_analytic)
        - [Alternative Method](#Experiment_Size_alternative)
    - [Dummy Tests (AA-Test)](#Dummy_Tests)
    - [Non-parametric Tests](#Non_parametric_Tests)
        - [When to use it?](#when_to_use_it)
        - [Types](#Types)
        - [A) Bootstrapping](#A_Bootstrapping)
        - [B) Permutation Tests](#B_Permutation_Tests)
        - [C) Rank-Sum Test (Mann-Whitney)](#C_Rank_Sum_Test)
        - [D) Sign Test](#D_Sign_Test)
    - [Analyzing Multiple Metrics](#Analyzing_Multiple_Metrics)
    - [Early Stopping](#Early_Stopping)
    

- [A/B Testing Case Study I](#A_B_Testing)
    - [Overview](#OverviewAB)
    - [Experiment I](#Experiment_I)
    - [Metric and Hypothesis](#Metric_and_Hypothesis)
    - [Experiment II](#Experiment_II)
    - [Multiple Tests](#Multiple_Tests)
    - [Difficulties in A/B Testing](#Difficulties_in_A_B_Testing)
    

- [A/B Testing Case Study II](#A_B_Testing_II)
    - [Defining the Goal](#Defining_the_Goal)
    - [Building a Funnel](#Building_a_Funnel)
    - [Deciding on Metrics](#Deciding_on_Metrics)
    - [Experiment Sizing](#Experiment_Sizing)
    - [Validity, Bias, and Ethics](#Validity_Bias_Ethics)
    - [Analyze Data](#Analyze_Data)
    - [Draw Conclusions](#Draw_Conclusions)
   
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
  ![image20]

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

- ***Funnel***: A funnel is the flow of steps you expect that a user of your product will take

Example: Online store
  - Visit the homepage
  - Search for a desired product or click on a product category
  - Click on a product image
  - Add the product to the cart
  - Check out and finalize purchase

  ![image6]

  One property to note about user funnels is that typically there will be some ***dropoff in the users*** that move from step to step. This is much like how an actual funnel narrows from a large opening to a small exit. Outside of an experiment, funnels can be used to analyze user flows. Observations from these flows can then be used to motivate experiments to try and improve the ***dropoff rates***.

  It's also worth noting that the flow through a funnel might be idealized compared to actual user practice. In the above example, users might perform multiple searches in a single session, or want to purchase multiple things. A user might access the site through a specific link, subverting the top part of the funnel. Refining the funnel and being specific about the kinds of events that are expected can help you create a consistent, reliable design and analysis.

- ***Unit of Diversion***: Way to assign users to either a control group or experimental group

  - ***Event-based diversion*** (e.g. pageview): Each time a user loads up the page of interest, the experimental condition is randomly rolled. Since this ignores previous visits, this can create an inconsistent experience, if the condition causes a user-visible change.
  - ***Cookie-based diversion***: A cookie is stored on the user's device, which determines their experimental condition as long as the cookie remains on the device. Cookies don't require a user to have an account or be logged in, but can be subverted through anonymous browsing or a user just clearing out cookies.
  - ***Account-based diversion*** (e.g. User ID): User IDs are randomly divided into conditions. Account-based diversions are reliable, but requires users to have accounts and be logged in. This means that our pool of data might be limited in scope, and you'll need to consider the risks of using personally-identifiable information.


  ![image7]

    In tis example Unit of Diversion = cookies

- ***Evaluation Metrics***:
    - metrics on which we will compare the two groups
    - these are the values that will tell us if our experiment is a success or not
    - we are not limited to just one metric, one can track multiple metrics

- Define also ***Invariant Metrics***:
    - metrics that do not change between groups
    - these metrics are useful to verify that control and experimental groups are comparable to one another except for for the experimental condition
    - e.g. number of cookies (each website version should get same number of visitors due to randomization)

    ![image8]


## Controlling variables <a name="Controlling_variables"></a>
[Correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation): Learn about potential problems that can weaken a study's usefulness
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

## Binomial Distribution <a name="Binomial_Distributioncs"></a>
- Notation 

    <img src="https://render.githubusercontent.com/render/math?math=X ~ B(n,p) = {n\choose k} \cdot p^{k} \cdot (1-p)^{n-k}" height="40px">

- Mean
    
    <img src="https://render.githubusercontent.com/render/math?math=\m =n \cdot p" height="20px">

- Standard Deviation
    
    <img src="https://render.githubusercontent.com/render/math?math=\m =\sqrt{n \cdot p \cdot (1 - p)}" height="50px">


## Computing p-value via stats.norm <a name="Computing_p_value"></a>
- Statmodels offers an easy approach to calculate p values

    'two-sided':
    ```
    p = 2 * stats.norm.cdf(-np.abs(z))
    ````
    'less'
    ```
    p = stats.norm.cdf(z)
    ```
    'greater'
    ```
    p = stats.norm.cdf(-z) = 1 - stats.norm.cdf(z)
    ```

## Statistical significance? <a name="Statistical_significance"></a>
- Let's say that we've collected data for a web-based experiment.
- Experiment: Test change in layout of a product information page 
- Goal: Higher revenue based on that feature
- Metric: cookie-based diversion
- Invariant metric: proportion of visitors assigned to one of our conditions, significance check via a two-sided test
- Evaluation metric: click-through rate, significance check via a one-sided test
- condition 0 -> control group
- condition 1 -> experimental group
- click 0 -> no click
- click 1 -> click


## Checking Invariant metric <a name="Checking_Invariant_metric"></a>
- Open notebook ```notebooks/Statistical_Significance.ipynb```

***A) Analytic Approach*** <a name="Analytic_Approach"></a>
- Use the exact binomial distribution (via mean, sd and z-score) to compute a p-value for the test
- Alternative: use the normal distribution approximation (possible due large sample size and the central limit theorem)
- To get a precise p-value, perform a continuity correction, either adding or subtracting 0.5 to the total count before computing the area underneath the curve
    ```
    # import packages
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    from statsmodels.stats import proportion as proptests
    import matplotlib.pyplot as plt
    % matplotlib inline

    # import data
    data = pd.read_csv('../data/statistical_significance_data.csv')
    data.head(10)

    # get number of trials 
    n_obs = data.shape[0]

    # get and number of 'successes'
    n_control = data.groupby('condition').size()[0]

    # Compute a z-score and p-value
    # probability of success
    p = 0.5

    # sd for binomial distribution
    sd = np.sqrt(p * (1-p) * n_obs)

    # z-score
    # to get a precise p-value, 0.5 = continuity correction, was added the total count before computing the area underneath the curve
    z = ((n_control + 0.5) - p * n_obs) / sd
    print(z)

    # p-value for two-sided test
    print(2 * stats.norm.cdf(z))

    Outcome:
    z = -0.506217597735
    p = 0.612703902554
    # We fail to reject the Null hypothesis
    ```

***B) Simulation Approach*** <a name="Simulation_Approach"></a>
- Simulate the number of visitors that would be assigned to each group via np.random.binomial samplimng
- assume, p=0.5, 200.000 repetitions
    ```
    # get number of trials and number of 'successes'
    n_obs = data.shape[0]
    n_control = data.groupby('condition').size()[0]

    # # simulate outcomes under null, compare to observed outcome
    p = 0.5
    n_trials = 200_000

    samples = np.random.binomial(n_obs, p, n_trials)

    print(np.logical_or(samples <= n_control, samples >= (n_obs - n_control)).mean())

    Outcome:
    0.611725
    ```

## Checking the Evaluation Metric <a name="Checking_the_Evaluation_Metric"></a>

- Open notebook ```notebooks/Statistical_Significance.ipynb```

    ```
    p_click = data.groupby('condition').mean()['click']
    p_click

    Outcome:
    condition
    0    0.079430
    1    0.112205
    Name: click, dtype: float64
    ```

    ```
    p_click[1] - p_click[0]
    Outcome:
    0.03277498917523293
    ```

***A) Analytic Approach*** <a name="Analytic_Approach_2"></a>
- Use the exact binomial distribution (via mean, sd and z-score) to compute a p-value for the test
- Computing the z-score and resulting p-value without a continuity correction 
- pooled click-through rate
- pooled standard deviation

    ```
    # get number of trials and overall 'success' rate under null
    n_control = data.groupby('condition').size()[0]
    n_exper = data.groupby('condition').size()[1]
    p_null = data['click'].mean()

    # compute standard error, z-score, and p-value
    se_p = np.sqrt(p_null * (1-p_null) * (1/n_control + 1/n_exper))

    z = (p_click[1] - p_click[0]) / se_p
    print(z)
    print(1-stats.norm.cdf(z))

    Outcome:
    z = 1.75718873962
    p = 0.0394428219746
    # We reject the Null: New feature is significant
    ``` 

***B) Simulation Approach*** <a name="Simulation_Approach_2"></a>
- Simulate the overall click-through rate that would be assigned to each group via np.random.binomial samplimng
- samples = overall click-through rate 

    ```
    # get number of trials and overall 'success' rate under null
    n_control = data.groupby('condition').size()[0]
    n_exper = data.groupby('condition').size()[1]
    p_null = data['click'].mean()

    # simulate outcomes under null, compare to observed outcome
    n_trials = 200_000

    ctrl_clicks = np.random.binomial(n_control, p_null, n_trials)
    exp_clicks = np.random.binomial(n_exper, p_null, n_trials)
    samples = exp_clicks / n_exper - ctrl_clicks / n_control

    print((samples >= (p_click[1] - p_click[0])).mean())

    Outcome:
    p = 0.039785
    # We reject the Null: New feature is significant
    ```

## Practical statistical significance? <a name="Practical_statistical_significance"></a>
- Even there is statistical significance of a new feature, it does not mean that it is worth to implement the new feature. 
- The new feature must also show practical significance.
- Practical significance refers to the level of effect that you need to observe for a true experimental success.

    ![image21]



- If you consider the confidence interval for an evaluation metric statistic against the null baseline and practical significance bound, there are a few cases that can come about.

    - ***A) Confidence interval is fully in practical significance region***

        Below, m0 indicates the null statistic value, dmin the practical significance bound, and the blue line the confidence interval for the observed statistic. We assume that we're looking for a positive change, ignoring the negative equivalent for dmin.

        If the confidence interval for the statistic does not include the null or the practical significance level, then the experimental manipulation can be concluded to ***have a statistically and practically significant effect***. It is clearest in this case that the manipulation should be implemented as a success.
    
    - ***B) Confidence interval completely excludes any part of practical significance region***

        If the confidence interval does not include any values that would be considered practically significant, this is a clear case for us to ***not implement the experimental change***. This includes the case where the metric is statistically significant, but whose interval does not extend past the practical significance bounds. With such a low chance of practical significance being achieved on the metric, we should be wary of implementing the change.

    - ***C) Confidence interval includes points both inside and outside practical significance bounds*** 

        This leaves the trickiest cases to consider, where the confidence interval straddles the practical significance bound. In each of these cases, there is an uncertain possibility of practical significance being achieved. In an ideal world, you would be able to collect more data to reduce our ***uncertainty***, reducing the scenario to one of the previous cases. Outside of this, you'll need to consider the risks carefully in order to make a recommendation on whether or not to follow through with a tested change. Your analysis might also reveal subsets of the population or aspects of the manipulation that do work, in order to refine further studies or experiments.

    ![image22]



## Experiment Size <a name="Experiment_Size"></a>
- Check this helpful [link](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm) 
- Practical significance boundaries can be used to plan an experiment.
- By knowing how many observations we need in order to detect our desired effect to our desired level of reliability, we can see how long we would need to run our experiment and whether or not it is feasible.
- Current click-trough-rate = 10%
- Practical significance boundary = 12% (desired click-through-rate)

    ![image23]


- Open notebook ```notebooks/Experiment_Size.ipynb``` 

    - The example explored below is a one-tailed test, with the alternative value greater than the null.
    - The power computations performed in the first part will not work if the alternative proportion is greater than the null, e.g. detecting a proportion parameter of 0.88 against a null of 0.9. You might want to try to rewrite the code to handle that case! The same issue should not show up for the second approach, where we directly compute the sample size
    - If we need to calculate a 'Two-tail test' then we need to split the significance (i.e. our alpha value) because we're still using a calculation method for one-tail. The split in half symbolizes the significance level being appropriated to both tails. A 95% significance level has a 5% alpha; splitting the 5% alpha across both tails returns 2.5%. Taking 2.5% from 100% returns 97.5% as an input for the significance level. 

    ***Method 1: Trial and Error*** <a name="Experiment_Size_trial_error"></a>
    ```
    def power(p_null, p_alt, n, alpha = .05, plot = True):
        """
        Compute the power of detecting the difference in two populations with 
        different proportion parameters, given a desired alpha rate.
        
        Input parameters:
            p_null: base success rate under null hypothesis
            p_alt : desired success rate to be detected, must be larger than
                    p_null
            n     : number of observations made in each group
            alpha : Type-I error rate
            plot  : boolean for whether or not a plot of distributions will be
                    created
        
        Output value:
            power : Power to detect the desired difference, under the null.
        """
        
        # Compute the power
        se_null = np.sqrt((p_null * (1-p_null) + p_null * (1-p_null)) / n)
        null_dist = stats.norm(loc = 0, scale = se_null)
        p_crit = null_dist.ppf(1 - alpha) # one-tailed test
        
        se_alt  = np.sqrt((p_null * (1-p_null) + p_alt  * (1-p_alt) ) / n)
        alt_dist = stats.norm(loc = p_alt - p_null, scale = se_alt)
        beta = alt_dist.cdf(p_crit)
        
        if plot:
            # Compute distribution heights
            low_bound = null_dist.ppf(.01)
            high_bound = alt_dist.ppf(.99)
            x = np.linspace(low_bound, high_bound, 201)
            y_null = null_dist.pdf(x)
            y_alt = alt_dist.pdf(x)

            # Plot the distributions
            plt.plot(x, y_null)
            plt.plot(x, y_alt)
            plt.vlines(p_crit, 0, np.amax([null_dist.pdf(p_crit), alt_dist.pdf(p_crit)]),
                    linestyles = '--')
            plt.fill_between(x, y_null, 0, where = (x >= p_crit), alpha = .5)
            plt.fill_between(x, y_alt , 0, where = (x <= p_crit), alpha = .5)
            
            plt.legend(['null','alt'])
            plt.xlabel('difference')
            plt.ylabel('density')
            plt.show()
        
        # return power
        return (1 - beta)
    
    power(.1, .12, 1000)

    Output:
    1-beta = 0.44122379261151545
    ```

    ***Method 2: Analytic Solution*** <a name="Experiment_Size_analytic"></a>
    ```
    def experiment_size(p_null, p_alt, alpha = .05, beta = .20):
        """
        Compute the minimum number of samples needed to achieve a desired power
        level for a given effect size.
        
        Input parameters:
            p_null: base success rate under null hypothesis
            p_alt : desired success rate to be detected
            alpha : Type-I error rate
            beta  : Type-II error rate
        
        Output value:
            n : Number of samples required for each group to obtain desired power
        """
        
        # Get necessary z-scores and standard deviations (@ 1 obs per group)
        z_null = stats.norm.ppf(1 - alpha) # one-tailed test
        z_alt  = stats.norm.ppf(beta) # one-tailed test
        sd_null = np.sqrt(p_null * (1-p_null) + p_null * (1-p_null))
        sd_alt  = np.sqrt(p_null * (1-p_null) + p_alt  * (1-p_alt) )
        
        # Compute and return minimum sample size
        p_diff = p_alt - p_null
        n = ((z_null*sd_null - z_alt*sd_alt) / p_diff) ** 2
        return np.ceil(n)
    
    experiment_size(.1, .12)

    Output:
    n = 2863.0
    ```

    ***Alternative Method*** <a name="Experiment_Size_alternative"></a>
    ```
    # example of using statsmodels for sample size calculation
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    # leave out the "nobs" parameter to solve for it
    NormalIndPower().solve_power(effect_size = proportion_effectsize(.12, .1), alpha = .05, power = 0.8, alternative = 'larger')

    Output:
    3020.515856462414
    ```

## Dummy Tests (AA-Test) <a name="Dummy_Tests"></a>
- In a dummy test, you will implement the same steps that you would in an actual experiment to assign the experimental units into groups. However, the experimental manipulation won't actually be implemented, and the groups will be treated equivalently.

    ![image24]

## Non-parametric Tests <a name="Non_parametric_Tests"></a>
- A parametric test uses a population’s parameters (for example, the mean or standard deviation)
- A [non parametric test](https://www.statisticshowto.com/parametric-and-non-parametric-data/) does not use info about the underlying distribution (for example, that the data comes from a normal distribution). 
- They can be used in a wider range of circumstances compared to standard tests
- Bootstrapping and Permutation Tests as resampling-type tests
- Rank-Sum and Sign Test as non resampling-type tests

### When to use it? <a name="when_to_use_it"></a>
- When your data isn’t normal
- As a second proof
- One or more assumptions of a parametric test have been violated
- Your sample size is too small to run a parametric test
- Your data has outliers that cannot be removed.
- You want to test for the median rather than the mean (you might want to do this if you have a very skewed distribution).

### Types <a name="Types"></a>
| Nonparametric Test     | Parametric Alternative     |
| :------------- | :------------- |
| Bootstrapping  |  all kinds possible |
|  Permutation Tests |   |
| 1-sample [sign test](https://www.statisticshowto.com/sign-test/)       |  	[One-sample Z-test](https://www.statisticshowto.com/one-sample-z-test/), One sample t-test       |
| 1-sample [Wilcoxon Signed Rank test](https://www.statisticshowto.com/wilcoxon-signed-rank-test/) |	[One-sample Z-test](https://www.statisticshowto.com/one-sample-z-test/), One sample t-test     |
| [Mann-Whitney test](https://www.statisticshowto.com/mann-whitney-u-test/) 	| [Independent samples t-test](https://www.statisticshowto.com/independent-samples-t-test/)  |

***A) Bootstrapping*** <a name="A_Bootstrapping"></a>
- A bootstrapped sample means drawing points from the original data with replacement until we get as many points as there were in the original data
- Taking a lot of bootstrapped samples allows us to estimate the sampling distribution for various statistics on our original data
- The bootstrap procedure is fairly simple and straightforward. Since you don't make assumptions about the distribution of data, it can be applicable for any case you encounter. The results should also be fairly comparable to standard tests
- For example, let's say that we wanted to create a 95% confidence interval for the 90th percentile from a dataset of 5000 data points. (Perhaps we're looking at website load times and want to reduce the worst cases.)

    To do so:
    - take a bootstrap sample (i.e., draw 5000 points with replacement from the original data), 
    - record the 90th percentile
    -  repeat this a large number of times, let's say 100 000. 
    - From this bunch of bootstrapped 90th percentile estimates, we form our confidence interval by finding the values that capture the central 95% of the estimates (cutting off 2.5% on each tail).

- Open notebook ```notebooks/Non-Parametric_Tests_Part_1.ipynb``` 

    ```
    def quantile_ci(data, q, c = .95, n_trials = 1000):
        """
        Compute a confidence interval for a quantile of a dataset using a bootstrap
        method.
        
        Input parameters:
            data: data in form of 1-D array-like (e.g. numpy array or Pandas series)
            q: quantile to be estimated, must be between 0 and 1
            c: confidence interval width
            n_trials: number of bootstrap samples to perform
        
        Output value:
            ci: Tuple indicating lower and upper bounds of bootstrapped
                confidence interval
        """
        
        # initialize storage of bootstrapped sample quantiles
        n_points = data.shape[0] # number of data points
        sample_qs = []# storage of sampled quantiles
        
        # For each trial...
        for _ in range(n_trials):
            # draw a random sample from the data with replacement...
            #sample = np.random.choice(data, size=n_points, replace=True)
            sample = data.sample(n_points, replace=True)
            
            # compute the desired quantile...
            #sample_q = np.percentile(sample, 100 * q)
            sample_q = sample.quantile(q=0.9)
            
            # and add the value to the list of sampled quantiles
            sample_qs.append(sample_q)
            
        # Compute the confidence interval bounds
        lower_limit = np.percentile(sample_qs, 2.5)
        upper_limit = np.percentile(sample_qs, 97.5)
        
        return (lower_limit, upper_limit)
    
    data = pd.read_csv('data/bootstrapping_data.csv')
    data.head(10)

    # data visualization
    plt.hist(data['time'], bins = np.arange(0, data['time'].max()+400, 400));

    lims = quantile_ci(data['time'], 0.9)
    rint(lims)

    Output:
    (5495.0, 5832.0024999999996)
    ```
***B) Permutation Tests*** <a name="B_Permutation_Tests"></a>
- The permutation test is a resampling-type test used to compare the values on an outcome variable between two or more groups.
- The idea here is that, under the null hypothesis, the outcome distribution should be the same for all groups, whether control or experimental. Thus, we can emulate the null by taking all of the data values as a single large group. Applying labels randomly to the data points (while maintaining the original group membership ratios) gives us one simulated outcome from the null.
- The rest is similar to the sampling approach used in a standard hypothesis test, 
- except that we haven't specified a reference distribution to sample from – we're sampling directly from the data we've collected. 
- Then apply the labels randomly to all the data 
- Record the outcome statistic many times
- Compare observed statistic against the simulated statistics. 
- A p-value is obtained by seeing how many simulated statistic values are as or more extreme than the one actually observed
- Draw a conclusion 

- For example, try implementing a permutation test in the cells below to test if the 90th percentile of times is statistically significantly smaller for the experimental group, as compared to the control group:

    To do so:
    - Initialize an empty list to store the difference in sample quantiles as sample_diffs.
    - Create a loop for each trial:

        - First generate a permutation sample by randomly shuffling the data point labels. ([numpy.random.permutation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html) will be useful here.)
        - Then, compute the qth quantile of the data points that have been assigned to each group based on the permuted labels. Append the difference in quantiles to the sample_diffs list.

    - After gathering the quantile differences for permuted samples, compute the observed difference for the actual data. Then, compute a p-value from the number of permuted sample differences that are less than or greater than the observed difference, depending on the desired alternative hypothesis.
    
- Open notebook ```notebooks/Non-Parametric_Tests_Part_1.ipynb``` 
    ```
    def quantile_permtest(x, y, q, alternative = 'less', n_trials = 10_000):
        """
        Compute a confidence interval for a quantile of a dataset using a bootstrap
        method.
        
        Input parameters:
            x: 1-D array-like of data for independent / grouping feature as 0s and 1s
            y: 1-D array-like of data for dependent / output feature
            q: quantile to be estimated, must be between 0 and 1
            alternative: type of test to perform, {'less', 'greater'}
            n_trials: number of permutation trials to perform
        
        Output value:
            p: estimated p-value of test
        """
        
        
        # initialize storage of bootstrapped sample quantiles
        sample_diffs = []
        
        # For each trial...
        for _ in range(n_trials):
            # randomly permute the grouping labels
            labels = np.random.permutation(y)
            
            # compute the difference in quantiles
            cond_q = np.percentile(x[labels == 0], 100 * q)
            exp_q  = np.percentile(x[labels == 1], 100 * q)
            
            # and add the value to the list of sampled differences
            sample_diffs.append(exp_q - cond_q)
        
        # compute observed statistic
        cond_q = np.percentile(x[y == 0], 100 * q)
        exp_q  = np.percentile(x[y == 1], 100 * q)
        obs_diff = exp_q - cond_q
        
        # compute a p-value
        if alternative == 'less':
            hits = (sample_diffs <= obs_diff).sum()
        elif alternative == 'greater':
            hits = (sample_diffs >= obs_diff).sum()
        
        return (hits / n_trials)
    
    data = pd.read_csv('../data/permutation_data.csv')
    data.head(10)

    # data visualization
    bin_borders = np.arange(0, data['time'].max()+400, 400)
    plt.hist(data[data['condition'] == 0]['time'], alpha = 0.5, bins = bin_borders)
    plt.hist(data[data['condition'] == 1]['time'], alpha = 0.5, bins = bin_borders)
    plt.legend(labels = ['control', 'experiment']);

    # Just how different are the two distributions' 90th percentiles?
    print(np.percentile(data[data['condition'] == 0]['time'], 90),
        np.percentile(data[data['condition'] == 1]['time'], 90))
    
    quantile_permtest(data['time'], data['condition'], 0.9, alternative = 'less')
    ```

***C) Rank-Sum Test (Mann-Whitney)*** <a name="C_Rank_Sum_Test"></a>
- Use this test to compare differences between two independent groups when dependent variables are either ordinal or continuous.
- ***The null hypothesis*** is that the two samples come from the same population (i.e. that they both have the same median) 
- ***The alternative hypothesis*** is that the probability is 50% that a randomly drawn member of the first population will exceed a member of the second population. 
- A very general formulation is to assume that:

    - All the observations from both groups are independent of each other,
    - The responses are ordinal (i.e., one can at least say, of any two observations, which is the greater),
    

    - <img src="https://render.githubusercontent.com/render/math?math=\mu_{U} = \frac{n_{1}n_{2}}{2}" width="100px">
    - <img src="https://render.githubusercontent.com/render/math?math=\sigma_{U} = \sqrt{\frac{n_{1}n_{2}(n_{1} %2B n_{2} %2B 1)}{12}}" width="200px">
    - <img src="https://render.githubusercontent.com/render/math?math=z = \frac{U - \mu_{U}}{\sigma_{U}}" width="100px">

    - For large samples, U is approximately normally distributed
    - Check also scipy stats package [mannwhitneyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)

- Algorithm for computing thge U-statistic:

    1. Name the sample with the smaller ranks “sample 1” and the sample with the larger ranks “sample 2”. Choosing the sample with the smaller ranks to be “sample 1” is optional, but it makes the computation easier.
    2. Take the first observation in sample 1. Count how many observations in sample 2 are smaller than it. If the observations are equal, count it as one half. For example, if you have ten that are less and two that are equal: 10 + 2(1/2) = 11.
    3. Repeat Step 2 for all observations in sample 1.
    4. Add up all of your totals from Steps 2 and 3. This is the U statistic.


- Open notebook ```notebooks/Non-Parametric_Tests_Part_2.ipynb``` 
    ```
    import numpy as np
    import pandas as pd
    import scipy.stats as stats

    import matplotlib.pyplot as plt
    % matplotlib inline

    def ranked_sum(x, y, alternative = 'two-sided'):
        """
        Return a p-value for a ranked-sum test, assuming no ties.
        
        Input parameters:
            x: 1-D array-like of data for first group
            y: 1-D array-like of data for second group
            alternative: type of test to perform, {'two-sided', less', 'greater'}
        
        Output value:
            p: estimated p-value of test
        """
        
        # compute U
        u = 0
        for i in x:
            wins = (i > y).sum()
            ties = (i == y).sum()
            u += wins + 0.5 * ties
        
        # compute a z-score
        n_1 = x.shape[0]
        n_2 = y.shape[0]
        mean_u = n_1 * n_2 / 2
        sd_u = np.sqrt( n_1 * n_2 * (n_1 + n_2 + 1) / 12 )
        z = (u - mean_u) / sd_u
        
        # compute a p-value
        if alternative == 'two-sided':
            p = 2 * stats.norm.cdf(-np.abs(z))
        if alternative == 'less':
            p = stats.norm.cdf(z)
        elif alternative == 'greater':
            p = stats.norm.cdf(-z)
        
        return p

    data = pd.read_csv('data/permutation_data.csv')
    data.head(10)

    # data visualization
    bin_borders = np.arange(0, data['time'].max()+400, 400)
    plt.hist(data[data['condition'] == 0]['time'], alpha = 0.5, bins = bin_borders)
    plt.hist(data[data['condition'] == 1]['time'], alpha = 0.5, bins = bin_borders)
    plt.legend(labels = ['control', 'experiment']);

    ranked_sum(data[data['condition'] == 0]['time'],
           data[data['condition'] == 1]['time'],
           alternative = 'greater')

    Output:
    0.0017522265022961059
    # We have evidence to reject the Null hypothesis
    ```

***D) Sign Test*** <a name="D_Sign_Test"></a>
- The sign test compares the sizes of two groups.
- SIze differences between groups are not important, only which group takes a larger value. 
- So comparisons of 0.21 vs. 0.22 and 0.21 vs. 0.31 are both counted equally as a point in favor of the second group. 
- A fairly weak test
- Use this test to estimate the median of a population and compare it to a reference value or target value.
- Most useful when only a few observations given and underlying distribution characteristics is not clear. 
- For example, you might use a sign test as an additional check on click rates that have been aggregated on a daily basis.
- The count of victories for a particular group can be modeled with the binomial distribution. 
- The data should be from two samples.
- The two dependent samples should be paired or matched. For example, depression scores from before a medical procedure and after.
    - H0: No difference in median of the signed differences.
    - H1: Median of the signed differences is less than zero.
- Open notebook ```notebooks/Non-Parametric_Tests_Part_2.ipynb``` 
    ```
    import numpy as np
    import pandas as pd
    import scipy.stats as stats

    import matplotlib.pyplot as plt
    % matplotlib inline

    def sign_test(x, y, alternative = 'two-sided'):
        """
        Return a p-value for a ranked-sum test, assuming no ties.
        
        Input parameters:
            x: 1-D array-like of data for first group
            y: 1-D array-like of data for second group
            alternative: type of test to perform, {'two-sided', less', 'greater'}
        
        Output value:
            p: estimated p-value of test
        """
        
        # compute parameters for Binominal distribution, subtract events where there is no value difference
        n = x.shape[0] - (x == y).sum()
        k = (x > y).sum() - (x == y).sum()

        # compute a p-value
        if alternative == 'two-sided':
            p = min(1, 2 * stats.binom(n, 0.5).cdf(min(k, n-k)))
        if alternative == 'less':
            p = stats.binom(n, 0.5).cdf(k)
        elif alternative == 'greater':
            p = stats.binom(n, 0.5).cdf(n-k)
        
        return p
    
    data = pd.read_csv('data/signtest_data.csv')
    data.head()

    # data visualization
    plt.plot(data['day'], data['control'])
    plt.plot(data['day'], data['exp'])
    plt.legend()

    plt.xlabel('Day of Experiment')
    plt.ylabel('Success rate');

    sign_test(data['control'], data['exp'], 'less')

    Output:
    0.089782714843750014
    ```
## Analyzing Multiple Metrics <a name="Analyzing_Multiple_Metrics"></a>
- Two metrics, each set up with a 5% Type I error rate -> chance for a falsely declared significant effect: 0.0975
    
    ![image25]

Bonferoni correction
- <img src="https://render.githubusercontent.com/render/math?math=\alpha_{ind} = \frac{\alpha_{over}}{n}" width="150px">

 Šidák correction
 - <img src="https://render.githubusercontent.com/render/math?math=\alpha_{ind} = 1 - (1- \alpha_{over})^{1/n}" width="290px">

    ![image26]

## Early Stopping <a name="Early_Stopping"></a>
- If you stop experiments early before data collection is complete (because the test is showing statistical significance), you run the risk of a significant increase in your Type I error rate

    ![image27]

- Open notebook ```notebooks/Early_Stopping.ipynb``` 

# A/B Testing Case Study I <a name="A_B_Testing"></a>
## Overview <a name="OverviewAB"></a>
- A/B tests are used to test changes on a web page by running an experiment where a control group sees the old version, while the experiment group sees the new version. 
- A metric is then chosen to measure the level of engagement from users in each group. 
- These results are then used to judge whether one version is more effective than the other. 
-A/B testing is very much like hypothesis testing with the following hypotheses:
    - Null Hypothesis: The new version is no better, or even worse, than the old version
    - Alternative Hypothesis: The new version is better than the old version

    ![image16]

## Experiment I <a name="Experiment_I"></a>
- The first change Audacity wants to try is on their homepage. They hope that this new, more engaging design will increase the number of users that explore their courses, that is, move on to the second stage of the funnel. 

    ![image18]

## Metric and Hypothesis <a name="Metric_and_Hypothesis"></a>
- Metric = feature that provide an objective measure of the success of an experimental manipulation
- e.g. Click Through Rate
- Here's the customer funnel for typical new users on their site:

    View home page > Explore courses > View course overview page > Enroll in course > Complete course

    ![image17]

- Open notebook under ```notebooks/Homepage Experiment Data.ipynb```



## Experiment II <a name="Experiment_II"></a>
- The second change Audacity is A/B testing is a more career focused description on a course overview page.
- They hope that this change may encourage more users to enroll and complete this course. 
- In this experiment, we’re going to analyze the following metrics:

    - Enrollment Rate: Click through rate for the Enroll button the course overview page
    - Average Reading Duration: Average number of seconds spent on the course overview page
    - Average Classroom Time: Average number of days spent in the classroom for students enrolled in the course
    - Completion Rate: Course completion rate for students enrolled in the course

    ![image19]

    - Open notebook under ```notebooks/enrollment_rate.ipynb```
    ```
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    % matplotlib inline

    np.random.seed(42)

    df = pd.read_csv('course_page_actions.csv')
    df.head()

    # Get dataframe with all records from control group
    control_df = df.query('group == "control"')

    # Compute click through rate for control group
    control_ctr = control_df.query('action == "enroll"').id.nunique() / control_df.query('action == "view"').id.nunique()

    # Display click through rate
    control_ctr

    # Get dataframe with all records from control group
    experiment_df = df.query('group == "experiment"')

    # Compute click through rate for experiment group
    experiment_ctr = experiment_df.query('action == "enroll"').id.nunique() / experiment_df.query('action == "view"').id.nunique()

    # Display click through rate
    experiment_ctr

    # Compute the observed difference in click through rates
    obs_diff = experiment_ctr - control_ctr

    # Display observed difference
    obs_diff

    # Create a sampling distribution of the difference in proportions
    # with bootstrapping
    diffs = []
    size = df.shape[0]
    for _ in range(10000):
        b_samp = df.sample(size, replace=True)
        control_df = b_samp.query('group == "control"')
        experiment_df = b_samp.query('group == "experiment"')
        control_ctr = control_df.query('action == "enroll"').id.nunique() / control_df.query('action == "view"').id.nunique()
        experiment_ctr = experiment_df.query('action == "enroll"').id.nunique() / experiment_df.query('action == "view"').id.nunique()
        diffs.append(experiment_ctr - control_ctr)

    # Convert to numpy array
    diffs = np.array(diffs)
    # Plot sampling distribution
    plt.hist(diffs);

    # Simulate distribution under the null hypothesis
    null_vals = np.random.normal(0, diffs.std(), diffs.size)
    # Plot the null distribution
    plt.hist(null_vals);

    # Plot observed statistic with the null distibution
    plt.hist(null_vals);
    plt.axvline(obs_diff, c='red')

    # Compute p-value
    (null_vals > obs_diff).mean()

    # Result: We have evidence to reject the Null hypothesis
    ```

For further metrics evaluation check:
- Open notebook under ```notebooks/average_classroom_time.ipynb```
- Open notebook under ```notebooks/completion_rate.ipynb```

- Result: Completion Rate - the null cannot be rejected --> Problem? --> Bonferroni Correction

## Multiple Tests - Multiple Metrics - Bonferroni Correction <a name="Multiple_Tests"></a>
- The Bonferroni Correction is one way to handle experiments with multiple tests, or metrics in this case. To compute the new bonferroni correct alpha value, we need to divide the original alpha value by the number of tests.

 - <img src="https://render.githubusercontent.com/render/math?math=\alpha_{new} = \frac{\alpha}{n}" width="150px">

- Since the Bonferroni method is ***too conservative*** when we expect correlation among metrics, we can better approach this problem with more sophisticated methods, such as the [closed testing procedure](https://en.wikipedia.org/wiki/Closed_testing_procedure), [Boole-Bonferroni bound](https://en.wikipedia.org/wiki/Boole%27s_inequality), and the [Holm-Bonferroni method](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method). These are less conservative and take this correlation into account.

- If you do choose to use a less conservative method, just make sure the assumptions of that method are truly met in your situation, and that you're not just trying to [cheat on a p-value](https://freakonometrics.hypotheses.org/19817). Choosing a poorly suited test just to get significant results will only lead to misguided decisions that harm your company's performance in the long run.

## Difficulties in A/B Testing <a name="Difficulties_in_A_B_Testing"></a>
- As one can see in the scenarios above, there are many factors to consider when designing an A/B test and drawing conclusions based on its results. To conclude, here are some common ones to consider.

    - ***Novelty effect*** and ***change aversion*** when existing users first experience a change
    - ***Sufficient traffic*** and conversions to have significant and repeatable results
    - ***Best metric choice*** for making the ultimate decision (eg. measuring revenue vs. clicks)
    - ***Long enough run time*** for the experiment to account for changes in behavior based on time of day/week or seasonal events.
    - ***Practical significance*** of a conversion rate (the cost of launching a new feature vs. the gain from the increase in conversion)
    - ***Consistency*** among test subjects in the control and experiment group (imbalance in the population represented in each group can lead to situations like [Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox))





# A/B Testing Case Study II <a name="A_B_Testing_II"></a>

Setting: 
- Fictional productivity software company 
- Goal: increase revenue. 
- At the moment: Software can be downloaded free of charge, for a 7-day trial. After the end of the trial pay for a license 
- Idea: change the layout of the homepage to make software more attractive. More downloads more potential purchases 

Work to do:
- Funnel
- Define metrics to track
- Experiment sizing
- Perform statistical tests
- Conclusions wrt the effectiveness of the new homepage

## Defining the Goal <a name="Defining_the_Goal"></a>
Before we do anything else, the first thing we should do is specify the objective or goal of our study:

    Optimizing homepage layout will increase the number of users and hence purchases

## Building a Funnel <a name="Building_a_Funnel"></a>
= activities that a user will take on the site that are relevant to measuring the success (wrt the goal)

- A straightforward flow might include the following steps:

    1. Visit homepage
    2. Visit download page
    3. Sign up for an account
    4. Download software
    5. After 7-day trial, software takes user to license-purchase page
    6. Purchase license

- Possible problems:
    - User might drop from the flow after each step

- Atypical events: 
    - might be users on the homepage who aren't new users 
    - Users don't come back after exactly seven days. Some come back earlier or some later

## Deciding on Metrics <a name="Deciding_on_Metrics"></a>
- Based on the Funnel 
    
    - Choose ***unit of diversion***: the point at which we divide observations into control and experiment groups groups
    - This can affect metrics as well
    - What metrics to track the ***success*** of the experimental manipulation
    - Define ***invariant*** and ***evaluation metrics***

- A ***cookie-based diversion*** is best for control and experimental group splitting.
    - We can split randomly visitors on their ***initial visit*** 
    - it's fairly reliable for tracking

- ***Metrics***: keep track of the number of cookies that are recorded in different parts of the website
    - number of cookies on the homepage (absolute count) 
    - number of cookies on the download page (absolute count)
    - number of cookies on the account registration page (absolute count)
    - the number of licenses purchased through the user accounts (absolute count)
    - software includes usage statistics that we could track
    - proportion of downloads out of all homepage visits (ratio)
    - License purchases vs. number of registered users, downloads or number of cookies (ratio)

- ***Invariant metrics*** are measures that are equal in control and experimental groups
- ***Evaluation metrics*** are measures are different in both groups and measure experimental manipulation

    ![image28]

## Experiment Sizing <a name="Experiment_Sizing"></a>
- take a look at the running time of the experiment


    Historical data
    - 3250 unique visitors per day, with slightly more visitors on Friday through Monday, than the rest of the week. 
    - There are about 520 software downloads per day (a .16 rate) 
    - 65 licenses purchased each day (a .02 rate). 
    - In an ideal case, both the download rate and license purchase rate should increase with the new homepage; 
    - A statistically significant negative change should be a sign to not deploy the homepage change. 
    - However, if only one of our metrics shows a statistically significant positive change we should be happy enough to deploy the new homepage.
    - Bonferonni correction is needed due to multiple testing


    Goal: 
    - detect an increase of 50 downloads per day (up to 570 per day, or a .175 rate). 
    - How many days do we need to collect data at 5% Type I error rate and at 80% power?

    Calculation:
    - start rate = 0.16
    - goal rate  = 0.175
    - Bonferonni correction -> 2 metrics -> alpha = 0.025
    ```
    # example of using statsmodels for sample size calculation
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    # leave out the "nobs" parameter to solve for it
    NormalIndPower().solve_power(effect_size = proportion_effectsize(.175, .16), alpha = .025, power = 0.8,
                                alternative = 'larger')
    Output: 
    9724 visitors
    ```
    - visitors per day in each group ~ 3250/2 = 1625
    - days needed = 9724 / 1625 ~ 6


    Possible problem:

    - Time between download and purchase a license about seven days.

    Solution

    - Add approximately seven days of safety

## Validity, Bias, and Ethics <a name="Validity_Bias_Ethics"></a>

Validity:
- ***Construct validity***: the evaluation metrics are directly aligned with the experimental goals
- ***Internal validity***: is maintained by performing an experiment with properly-handled randomization
- ***External validity***: given, as we are drawing from the full site population and there's no other population involved

Bias:
- ***novelty bias*** as being a potential issue. However, we don't expect users to come back to the homepage regularly. Downloading and license purchasing are actions we expect to only occur once per user, so there's no real 'return rate' to worry about. 

Ethic issues:
- A standard popup to let visitors know that cookies are used to track user experience on the site will likely suffice. 
- The largest ethics principle we should be concerned about is data sensitivity.
- No sensitive data is required for the metrics, 
- Do evaluations without investigaing any individual outcomes.

## Analyze Data <a name="Analyze_Data"></a>
- data was collected for 29 days
- daily counts for 
    - The number of unique cookies 
    - Number of downloads
    - Number of license purchases 
- daily counts are attributed to each group: the experimental group with the new homepage, or the control group with the old homepage

- Check the invariant metric
    - The number of cookies assigned to each group
    - If there is a statistically significant difference detected, then we shouldn't move on to the evaluation metrics right away
    - Possible issues: problem with the group-assignment procedure, or with the cookies

- Evaluation Metrics
    - Assuming that the invariant metric passed inspection, 
    - Download rate (total number of downloads divided by the number of cookies)
    - License purchasing rate (number of licenses divided by the number of cookies)
     
## Draw Conclusions <a name="Draw_Conclusions"></a>













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
* [17 Statistical Hypothesis Tests in Python ](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/)

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
