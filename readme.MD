### CS 462 - Assignment 4. Text Clustering and Decision Trees

#### Due Wednesday April 3, 11:59pm. 

_I have added optional milestone recommendations for each assignment to help you organize your time. These are just suggestions._

#### Part 1: using sklearn to do text classification. 

In this question, you'll be mostly working with existing code in sklearn. To begin, please work through [this tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
A modified version of the code is in document_classification.py

**(5 points)** _(Tues Mar 19)_ Add metrics into document_classification.py

**(5 points)** _(Wed Mar 20)_ Add parameter tuning to document_classification.py.

**(10 points)** _(Thu Mar 21)_ Modify document_classification.py to use the movie reviews dataset. Measure your performance using the sklearn cross_val_score function.

#### Part 2. using sklearn to do clustering.

In this section, you'll see how to use sklearn to do K-means clustering. There's no code to write here, but you will need
to run the algorithm with different paramters.

To begin, please work through [this tutorial](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html).

The modified code is in kmeans.py.
I've started with all 20 newsgroups being used as input to the clustering algorithm. This is pretty difficult (we think). 
In this question, you'll compare different inputs and their results on performance. 

**(10 points)** _(Fri Mar 22)_ For each of the following combinations, run kmeans and measure homogeneity and completeness. Also find the top five terms in that
cluster (see the tutorial to see how to do this). Prepare a table with your results.

- Choose two very dissimilar groups

- Choose two very similar groups. 

- Choose four very dissimilar groups

- Choose 4 similar groups. 

- Choose all 20 groups.

#### Part 3: Decision tree 

This is the largest part of the assignment - we'll implement the basic decision tree algorithm. 
I've provided two starter datasets (tennis and restaurant) for you, plus some unit tests.
Start with these datasets, and then test on breast cancer once you've got it working.


**(5 points)** _(Fri Mar 22)_ I've implemented entropy for you. 

Use that to implement gain. I have provided a unit test for you. 

**(5 points)** _(Mon Mar 25)_ implement split information and information gain ratio. I've provided a unit test for split information; add a unit test for information_gain_ratio.

**(10 points)**  _(Mon Mar 25)_ Use information_gain_ratio to implement select_attribute. Also add a unit test for select_attribute.

**(10 points)**  _(Wed Mar 27)_ Use select_attribute to implement make_tree. Add a unit test for make_tree.


**(10 points)**  _(Fri Mar 29)_ Now you are ready to implement classify. Add a unit test for this as well.

**(10 points)** _(Mon Apr 1)_ Next, measure the performance of your tree on both the restaurant and the tennis data using five-fold cross-validation. You can use either your own implementation from assignment 3 or the sklearn framework.

**(5 points)**  _(Mon April 1)_ Once that's working, try your tree out on the breast cancer data. Again, measure the performance using five-fold cross-validation.


#### Part 4: Knowledge Graphs 

Wikipedia sits on top of a knowledge graph (called WikiData). In this question, you'll get a chance to explore this knowledge graph.

To begin, open the page for [the Perseverance rover](https://en.wikipedia.org/wiki/Perseverance_(rover)). Go to 'tools' (on the right) and then WikiData to bring up the data page.

Every WikiData object has a unique id - this one is Q87749354. 

Scroll down to see some of the statements associated with this object. 
Note that each of the relations is also an object. For example, click on 'video' to learn what this link means.

We can search this graph in one of two ways: using a query language called SPARQL, or with a web-based tool. 
We'll do the latter, but if you're familiar with SQL, check out SPARQL.

**(10 points)** _(Tues April 3)_ To start, go to the [Wikipedia Query Builder](https://query.wikidata.org/querybuilder/?uselang=en) and enter "instance of" and "Mars Rover" in the Property and Value boxes.
How many results come up? What are they?

Now add a second condition, that it must be a part of the Mars 2020 mission. Now how many results do you get?

**(10 points)** _(Tues April 3)_ Now let's visualize this. Click on the 'graph' icon (it looks like arrows in a loop) and then select the Perseverance node. Notice that you can expand the nodes that are connected to Perseverance to answer questions.
Use this to determine the answers to the following questions:
- What type of space launch vehicle carried Perseverance to Mars, and what country built it?
- Who was the launch contractor for the Mars 2020 mission? What companies own them?
- what mission does Perseverance follow? Where did it land?
- What powers Perseverance? What two organizations designed it?

Grad students only:
Regulation of AI is a popular and pressing concern for world governments. The European Union has recently passed legislation to address
this, in the form of the EU AI Act. [This article](https://aimagazine.com/ai-strategy/eu-ai-act-regulating-techs-future-as-world-first-laws-pass) summarizes it, and the entire text can be found [here].(https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)

Please look at this material and answer the following questions:
1. The Act defines four levels of risk. What are they? What's an example of each class of risk?
2. What must providers of general (or foundation) models such as Gemini and GPT-4 provide?
3. Does the EU AI Act allow for real-time use of biometric information?







