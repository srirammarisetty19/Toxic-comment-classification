---
layout: post
title: 'Toxic Comments Classification'
date: '2018-12-29 21:21:21'
tags: ML, classification
---


<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. Problem Statement:</a>
<ul>
<li><a href="#sec-1-1">1.1. Introduction:</a></li>
<li><a href="#sec-1-2">1.2. Problem:</a></li>
</ul>
</li>
<li><a href="#sec-2">2. Motivation:</a></li>
<li><a href="#sec-3">3. Dataset:</a>
<ul>
<li><a href="#sec-3-1">3.1. Data Resource:</a></li>
<li><a href="#sec-3-2">3.2. Data Overview:</a></li>
</ul>
</li>
<li><a href="#sec-4">4. Approach:</a>
<ul>
<li><a href="#sec-4-1">4.1. How probability was calculated?</a></li>
</ul>
</li>
<li><a href="#sec-5">5. Analysis of Dataset:</a>
<ul>
<li><a href="#sec-5-1">5.1. Visualization</a>
<ul>
<li><a href="#sec-5-1-1">5.1.1. Count of number of comments in each class.</a></li>
<li><a href="#sec-5-1-2">5.1.2. Pie chart of Label Distribution over comments(without "none" category).</a></li>
<li><a href="#sec-5-1-3">5.1.3. Count for each label combination.</a></li>
<li><a href="#sec-5-1-4">5.1.4. Correlation matrix.</a></li>
<li><a href="#sec-5-1-5">5.1.5. Digging more into correlations using Venn Diagrams.</a></li>
<li><a href="#sec-5-1-6">5.1.6. Word analysis for each cateogires.</a></li>
</ul>
</li>
</ul>
</li>
<li><a href="#sec-6">6. Feature Engineering</a>
<ul>
<li><a href="#sec-6-1">6.1. Cleaning the comments</a></li>
<li><a href="#sec-6-2">6.2. Stemmers and Lemmatizers</a></li>
<li><a href="#sec-6-3">6.3. Vectorization</a></li>
<li><a href="#sec-6-4">6.4. Adding data related features</a></li>
</ul>
</li>
<li><a href="#sec-7">7. Model Building</a></li>
<li><a href="#sec-8">8. Training, Validation and Test Metrics.</a>
<ul>
<li><a href="#sec-8-1">8.1. Training and Validation split:</a></li>
<li><a href="#sec-8-2">8.2. Test Metric:</a></li>
<li><a href="#sec-8-3">8.3. Results for various models.</a>
<ul>
<li><a href="#sec-8-3-1">8.3.1. Base model:</a></li>
<li><a href="#sec-8-3-2">8.3.2. Random Forest:</a></li>
<li><a href="#sec-8-3-3">8.3.3. Logistic Regression:</a></li>
</ul>
</li>
</ul>
</li>
<li><a href="#sec-9">9. Sources And References.</a></li>
<li><a href="#sec-10">10. Conclusion</a></li>
</ul>
</div>
</div>


# Problem Statement:<a id="sec-1" name="sec-1"></a>

## Introduction:<a id="sec-1-1" name="sec-1-1"></a>

Discussing things you care about can be difficult. The threat of abuse
and harrasment online means that many people stop experssing themselves
and give up on seeking different opinions. Platforms struggle to 
efficiently facilitate conversations, leading many communities to limit
or completely shut down user comments.

## Problem:<a id="sec-1-2" name="sec-1-2"></a>

Building a multi-headed model that's capable of detecting different types
of toxicity like threats, obscenity, insult and identity-based hate.   

# Motivation:<a id="sec-2" name="sec-2"></a>

So far we have a range of publicly available models served through the
[Perspective API](https://perspectiveapi.com/#/), including toxicity. But the current models still make
errors, and they don't allow users to select which type of toxicity
they're interested in finding. (e.g. some platforms may be fine with
profanity, but not with other types of toxic content)

# Dataset:<a id="sec-3" name="sec-3"></a>

## Data Resource:<a id="sec-3-1" name="sec-3-1"></a>

The dataset used was Wikipedia corpus dataset which was rated by human raters for toxicity. The corpus contains
comments from discussions relating to use pages and articles dating from 2004-2015. The dataset was hosted on
Kaggle.

## Data Overview:<a id="sec-3-2" name="sec-3-2"></a>

The comments were manually classified into following categories
-   toxic
-   severe\_toxic
-   obscene
-   threat
-   insult
-   identity\_hate

The Dataset had 150k comments which were classified into one or more above categories.
The problem was to predict the probabilities of 10k comments being classified into multiple categories.

# Approach:<a id="sec-4" name="sec-4"></a>

## How probability was calculated?<a id="sec-4-1" name="sec-4-1"></a>

Though there many multi class classifiers, we didn't find a suitable multi label classifier which was able
to give probability with which target belongs to a label.

So, we used [scikit-learn OneVsRestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) with various estimators, with the help of [predict\_proba](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier.predict_proba), we predicted the 
probability with which a comment belongs to a particular label.

# Analysis of Dataset:<a id="sec-5" name="sec-5"></a>

## Visualization<a id="sec-5-1" name="sec-5-1"></a>

### Count of number of comments in each class.<a id="sec-5-1-1" name="sec-5-1-1"></a>

![img](/assets/TCC/count.png)

The three major labels are:
-   toxic
-   obscene
-   insult

### Pie chart of Label Distribution over comments(without "none" category).<a id="sec-5-1-2" name="sec-5-1-2"></a>

![img](/assets/TCC/pie.png)

### Count for each label combination.<a id="sec-5-1-3" name="sec-5-1-3"></a>

Now, let's take a look at number of comment for each label combination.This 
helps us in finding correlation between categories. \newline

![img](/assets/TCC/cat_corr.png)

<span class="underline">Following can be inferred from above table:</span>

-   The table shows that number of comments with only **none** label are high.
-   **toxic** which is the lable high after **none** is present in all top 6 combinations.
-   Among the top combinations, **obscene** and **insult** comes 4 times in 6.
-   As the combinations increase (i.e, a comment belonging to more categories) the
    count drops very fast.

### Correlation matrix.<a id="sec-5-1-4" name="sec-5-1-4"></a>

![img](/assets/TCC/corr.png)

<span class="underline">Following can be inferred from above matrix:</span>
-   **Toxic** is highly correlated with **obscene** and **insult**
-   **Insult** and **obscene** have <span class="underline">**highest**</span> correlation factor of \(0.74\)

<span class="underline">Interesting things to be observed:</span>
-   Though a **severe toxic** comment is also a **Toxic** comment, the
    correlation between them is only <span class="underline">0.31</span>.

### Digging more into correlations using Venn Diagrams.<a id="sec-5-1-5" name="sec-5-1-5"></a>

![img](/assets/TCC/venn.png)

This venn diagram shows the interpretations inferred from the correlation matrix above.

![img](/assets/TCC/corr_2.png)

-   This venn diagram shows that if a comment is **severe toxic** it indeed is also a **toxic**
          comment.

-   The low correlation factor is explained by the fact that **severe toxic** represents a small
    percentage to **toxic**. This is similar to <span class="underline">**Simpson's Paradox**</span>.

### Word analysis for each cateogires.<a id="sec-5-1-6" name="sec-5-1-6"></a>

![img](/assets/TCC/none.png)

![img](/assets/TCC/toxic.png)

![img](/assets/TCC/severe_toxic.png)

![img](/assets/TCC/obscene.png)

![img](/assets/TCC/threat.png)

![img](/assets/TCC/insult.png)

![img](/assets/TCC/identity_hate.png)

The vocabulary used in all categories is quite similar.

# Feature Engineering<a id="sec-6" name="sec-6"></a>

## Cleaning the comments<a id="sec-6-1" name="sec-6-1"></a>

-   Since, the comments in the dataset were collected from the internet
    they may contain 'HTML' elements in them. So, we removed the HTML from
    the test by using BeautifulSoup.
    
        review_text = BeautifulSoup(raw_review).get_text()

-   We then converted each comment into lower case and then splitted it
    into individual words.
    
        words = review_text.lower().split()

-   There were some words in the dataset which had length > 100, since there
    are no words in the english language whose length > 100, we wrote a 
    function to remove such words.
    
        def remove_big_words(words):
        l = []
        for word in words:
            if len(word) <= 100:
                l.append(word)
        return l

-   First, we tried building the features removing stop words and then 
    trained some models thinking that it may help the model in learning
    the semantics of toxicity, but we found out that the model learns
    better if there are stop words in the comment.
    
    Possible reason is, generally a hate/toxic comment is used
    towards a person, seeing the data we found out that those persons
    are generally referred by pronouns, which are nothing but stop words.

## Stemmers and Lemmatizers<a id="sec-6-2" name="sec-6-2"></a>

1.  **Definitions**:

    -   **Stemming** usually refers to a crude heuristic process that chops off
        the ends of words in the hope of achieving this goal correctly most of
        the time, and often includes removal of derivational affixes.
    
    -   **Lemmatization** usually refers to doing things properly with the use of
        a vocabulary and morphological analysis of words, normally aiming to
        remove inflectional endings only and to return the base or dictionary
        form of a word, which is known as the lemma.
        
        (Definitions source: [stemming and Lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html))

2.  **Reasons to use**:

    -   We used both [snowball stemmer](https://www.nltk.org/_modules/nltk/stem/snowball.html), [porter stemmer](https://www.nltk.org/api/nltk.stem.html#module-nltk.stem.porter) and [wordnet lemmatizer](https://www.nltk.org/_modules/nltk/stem/wordnet.html).
    
    -   For gramatical reasons, documents are going to use different forms
        of a word, such as *organizes*, *organize* and *organizing*. But
        they all represent the same semantics. So, using stemmer/Lemmatizer
        for those three words gives a single word, which helps algorithm
        learn better.

3.  **Results**:

    -   On **public Dataset**: (Decreasing order of accuracy)
        
        Snowball Stemmer > WordNet Lemmatizer > Porter Stemmer
    
    -   On **private Dataset**: (Decreasing order of accuracy)
        
        WordNet Lemmatizer > Snowball Stemmer > Porter Stemmer

## Vectorization<a id="sec-6-3" name="sec-6-3"></a>

Python's scikit-learn deals with numeric data only. To conver the text data
into numerical form, *tf-idf* vectorizer is used. [TF-IDF](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) vectorizer converts
a collection of raw documents to a matrix of Tf-idf features.

We fit the predictor variable on the dataset with tf-idf vectorizer, in
two different ways. First, by setting the parameter analyzer as 'word'(select words)
and the second by setting it to 'char'(select characters). Using 'char' was
important because the data had many 'foreign languages' and they were difficult
to deal with by considering only the 'word' analyzer.

We set the paramater n-gram range (an n-gram is a continguous sequence of
n-items from a given sample of text or speech). After trying various values, we set the ngram as (1, 1) for
'word' analyzer and (1, 4) for 'char' analyzer. We also set the max\_features
as 30000 for both word and char analyzer after many trails.

We then combined the word and character features and transformed the dataset
into two sparse matrixes for train and test sets, respectively using tf-idf
vectorizer.

## Adding data related features<a id="sec-6-4" name="sec-6-4"></a>

We tried adding features to the dataset that are computed from the data itself.
Those features are:

1.  Length of comments
2.  Number of exclamation marks - Data showed severe toxic comments with multiple exclamation marks.
3.  Number of question marks
4.  Number of punctuation symbols - Assumption is that angry people might not use punctuation symbols.
5.  Number of symbols - there are some comments with words like f\*\*k, $#\*t etc.
6.  Number of words
7.  Number of unique words - Data showed that angry comments are sometimes repeated many times.
8.  Proportion of unique words

Correlation between above features and labels:

![img](/assets/TCC/corr_1_1.jpeg)

![img](/assets/TCC/corr_1_2.jpeg)

<span class="underline">**Conclusion**</span> from above correlation matrix:

All the above features had correlation of **<** 0.06 with all labels. So, we 
decided that adding these features doesn't give benefit the model.

# Model Building<a id="sec-7" name="sec-7"></a>

Our basic pipeline consisted of count vectorizer or a tf-idf vectorizer and
a classifier. We used OneVsRest Classifier model. We trained the model with
Logistic Regression (LR), Random Forest(RF) and Gradient Boosting(GB) classifiers.
Among them LR gave good probabilities with default parameters.

So, we then improved the LR model by changing its parameters. 

# Training, Validation and Test Metrics.<a id="sec-8" name="sec-8"></a>

## Training and Validation split:<a id="sec-8-1" name="sec-8-1"></a>

To know whether was generalizable or not, we divided the into 
train and validation sets in 80:20 ratio. We then trained 
various models on the training data, then we ran the models
on validation data and we checked whether the model is 
generalizable or not.

Also, we trained different models on training data and tested
those on validation data, then we arrived at our best model.

## Test Metric:<a id="sec-8-2" name="sec-8-2"></a>

We used Receiver Operating Characteristic(ROC) along with Area
under the curve(AUC) as test metric.

## Results for various models.<a id="sec-8-3" name="sec-8-3"></a>

### Base model:<a id="sec-8-3-1" name="sec-8-3-1"></a>

We created a model <span class="underline">without any preprocessing or parameter tuning</span>,
we used this model as our model, and measured our progress using 
this model.

For this we used Logistic Regression as Classifier.

1.  **<span class="underline">Cross Validation Results</span>**.

    <table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
    
    
    <colgroup>
    <col  class="left" />
    
    <col  class="right" />
    </colgroup>
    <thead>
    <tr>
    <th scope="col" class="left">Category</th>
    <th scope="col" class="right">CV Score</th>
    </tr>
    </thead>
    
    <tbody>
    <tr>
    <td class="left">Toxic</td>
    <td class="right">0.9501</td>
    </tr>
    
    
    <tr>
    <td class="left">Severe\_toxic</td>
    <td class="right">0.9795</td>
    </tr>
    
    
    <tr>
    <td class="left">Obscene</td>
    <td class="right">0.9709</td>
    </tr>
    
    
    <tr>
    <td class="left">Threat</td>
    <td class="right">0.9733</td>
    </tr>
    
    
    <tr>
    <td class="left">Insult</td>
    <td class="right">0.9608</td>
    </tr>
    
    
    <tr>
    <td class="left">Identity\_hate</td>
    <td class="right">0.9548</td>
    </tr>
    </tbody>
    
    <tbody>
    <tr>
    <td class="left">Average CV</td>
    <td class="right">0.9649</td>
    </tr>
    </tbody>
    </table>

2.  **<span class="underline">ROC-AUC Curve</span>**.

    ![img](/assets/TCC/base_all.png)

### Random Forest:<a id="sec-8-3-2" name="sec-8-3-2"></a>

Next, we created our model using Random Forest.
We used *n\_estimators = 10* and *random\_state = 1* as parameters.

We observed the following results

1.  **<span class="underline">Cross Validation Results</span>**.

    <table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
    
    
    <colgroup>
    <col  class="left" />
    
    <col  class="right" />
    </colgroup>
    <thead>
    <tr>
    <th scope="col" class="left">Category</th>
    <th scope="col" class="right">CV Score</th>
    </tr>
    </thead>
    
    <tbody>
    <tr>
    <td class="left">Toxic</td>
    <td class="right">0.8984</td>
    </tr>
    
    
    <tr>
    <td class="left">Severe\_toxic</td>
    <td class="right">0.8479</td>
    </tr>
    
    
    <tr>
    <td class="left">Obscene</td>
    <td class="right">0.9491</td>
    </tr>
    
    
    <tr>
    <td class="left">Threat</td>
    <td class="right">0.6816</td>
    </tr>
    
    
    <tr>
    <td class="left">Insult</td>
    <td class="right">0.9183</td>
    </tr>
    
    
    <tr>
    <td class="left">Identity\_hate</td>
    <td class="right">0.7782</td>
    </tr>
    </tbody>
    
    <tbody>
    <tr>
    <td class="left">Average CV</td>
    <td class="right">0.7782</td>
    </tr>
    </tbody>
    </table>

2.  **<span class="underline">ROC-AUC Curve</span>**.

    ![img](/assets/TCC/rf.png)
    
    From the Cross Validation results table and ROC-AUC Curve, its clear 
    that Random Forest performs poorly compared to our base model itself,
    So we proceeded to tune parameters for Logistic Regression for better
    accuracy.

### Logistic Regression:<a id="sec-8-3-3" name="sec-8-3-3"></a>

**<span class="underline">(I)</span>**

We created one model using *C = 4* as parameter.
The following results were observed.

1.  **<span class="underline">Cross Validation Results</span>**:

    <table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
    
    
    <colgroup>
    <col  class="left" />
    
    <col  class="right" />
    </colgroup>
    <thead>
    <tr>
    <th scope="col" class="left">Category</th>
    <th scope="col" class="right">CV Score</th>
    </tr>
    </thead>
    
    <tbody>
    <tr>
    <td class="left">Toxic</td>
    <td class="right">0.9690</td>
    </tr>
    
    
    <tr>
    <td class="left">Severe\_toxic</td>
    <td class="right">0.9850</td>
    </tr>
    
    
    <tr>
    <td class="left">Obscene</td>
    <td class="right">0.9825</td>
    </tr>
    
    
    <tr>
    <td class="left">Threat</td>
    <td class="right">0.9856</td>
    </tr>
    
    
    <tr>
    <td class="left">Insult</td>
    <td class="right">0.9750</td>
    </tr>
    
    
    <tr>
    <td class="left">Identity\_hate</td>
    <td class="right">0.9774</td>
    </tr>
    </tbody>
    
    <tbody>
    <tr>
    <td class="left">Average CV</td>
    <td class="right">0.9791</td>
    </tr>
    </tbody>
    </table>

2.  **<span class="underline">ROC-AUC Curve</span>**.

    ![img](/assets/TCC/c=4.png)
    
    **<span class="underline">(II)</span>**
    
    We created another Logistic Regression by selecting the best parameters
    by crossvalidating the following parameters.
    
    <table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
    
    
    <colgroup>
    <col  class="right" />
    
    <col  class="left" />
    
    <col  class="left" />
    
    <col  class="left" />
    </colgroup>
    <thead>
    <tr>
    <th scope="col" class="right">C</th>
    <th scope="col" class="left">fit\_intercept</th>
    <th scope="col" class="left">penalty</th>
    <th scope="col" class="left">class\_weight</th>
    </tr>
    </thead>
    
    <tbody>
    <tr>
    <td class="right">1.05</td>
    <td class="left">True</td>
    <td class="left">'l2'</td>
    <td class="left">None</td>
    </tr>
    
    
    <tr>
    <td class="right">0.2</td>
    <td class="left">True</td>
    <td class="left">'l2'</td>
    <td class="left">'balanced'</td>
    </tr>
    
    
    <tr>
    <td class="right">0.6</td>
    <td class="left">True</td>
    <td class="left">'l2'</td>
    <td class="left">'balanced'</td>
    </tr>
    
    
    <tr>
    <td class="right">0.25</td>
    <td class="left">True</td>
    <td class="left">'l2'</td>
    <td class="left">'balanced'</td>
    </tr>
    
    
    <tr>
    <td class="right">0.45</td>
    <td class="left">True</td>
    <td class="left">'l2'</td>
    <td class="left">'balanced'</td>
    </tr>
    
    
    <tr>
    <td class="right">0.25</td>
    <td class="left">True</td>
    <td class="left">'l2'</td>
    <td class="left">'balanced'</td>
    </tr>
    </tbody>
    </table>

3.  **<span class="underline">Cross Validation Results</span>**:

    <table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
    
    
    <colgroup>
    <col  class="left" />
    
    <col  class="right" />
    </colgroup>
    <thead>
    <tr>
    <th scope="col" class="left">Category</th>
    <th scope="col" class="right">CV Score</th>
    </tr>
    </thead>
    
    <tbody>
    <tr>
    <td class="left">Toxic</td>
    <td class="right">0.9675</td>
    </tr>
    
    
    <tr>
    <td class="left">Severe\_toxic</td>
    <td class="right">0.9864</td>
    </tr>
    
    
    <tr>
    <td class="left">Obscene</td>
    <td class="right">0.9827</td>
    </tr>
    
    
    <tr>
    <td class="left">Threat</td>
    <td class="right">0.9847</td>
    </tr>
    
    
    <tr>
    <td class="left">Insult</td>
    <td class="right">0.9761</td>
    </tr>
    
    
    <tr>
    <td class="left">Identity\_hate</td>
    <td class="right">0.9764</td>
    </tr>
    </tbody>
    
    <tbody>
    <tr>
    <td class="left">Average CV</td>
    <td class="right">0.9790</td>
    </tr>
    </tbody>
    </table>

4.  **<span class="underline">ROC-AUC Curve</span>**.

    ![img](/assets/TCC/c_parameters.png)
    
    <span class="underline">**Though**</span>, (I) gave better score compared to (II) on validation set,
     with difference in order of *0.0001*.
     when run on the acutal data (II) was found to better than (I)

# Sources And References.<a id="sec-9" name="sec-9"></a>

1.  <https://blog.citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics>
2.  <https://www.data-to-viz.com/>
3.  <https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda>

# Conclusion<a id="sec-10" name="sec-10"></a>

After checking the kaggle discussion board of the actual competition, standard Machine Learning 
approaches yield a maximum score of 0.9792, irrespective of any aprroach. In order to get a 
large margin over this score one has to employ Deep Learning(DL) techniques.
