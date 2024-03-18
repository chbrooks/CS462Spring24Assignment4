import pandas as pd
from math import log2 as log
import math

## assume that values is a pandas series: [True, False, True, False]
## return the entropy of this series

def entropy(values) :
    counts = values.value_counts()
    e = sum([item/sum(counts) * math.log(item/sum(counts),2) * -1 for item in counts])
    return e

## variables is a pandas Series containing attribute values: ['rainy','overcast','rainy','sunny']
## classifications is a pandas Series containing classifications. ['yes','no',yes',yes']

def gain(variables, classifications) :
    pass

## the split information for a list of categories is:
## sum(prob(c) * lg(prob(c) for c in categories)
def split_information(variables):
    pass

## variables is a pandas Series containing attribute values: ['rainy','overcast','rainy','sunny']
## classifications is a pandas Series containing classifications. ['yes','no',yes',yes']

def information_gain_ratio(variables, classifications) :
    pass


## df is a pandas dataframe, and classifications the corresponding
# classifications.
## check each column in the dataframe and return the column label
# of the column which maximizes gain (minimizes remainder.)

def select_attribute(df, classifications) :
  pass

## Here's our node class.
## leaf nodes contain a classification
## Non-leaf nodes contain an attribute, and a dictionary that maps
## attribute values to children (which are Nodes).

class Node :
    def __init__(self, classification=None,attribute=None):
        self.classification = classification
        self.attribute = attribute
        self.children = {}

    def isLeaf(self):
        return len(self.children) == 0

## This is a recursive function.
## Base case #1. Our data has 0 entropy. We are done. Create and return
## a leaf node containing the value stored in the (right-hand) classification
## column.
## Base case #2. We are out of rows. There is no more data.
# Call ZeroR on the whole dataset and use this value.
## Base Case #3 We are out of columns. There is noise in our data.
# Call ZeroR on the whole dataset and use this value.
## Recursive step: Call select_attribute to find the attribute that maximizes
## gain (or minimizes remainder).
## Then, split your dataset. For each value of that attribute, select the rows
## that contain that attribute value, and construct a subtree (removing the selected attribute)
## That subtree is added to the children dictionary.
## Question: How do you deal with subtrees for missing values?
## When setting up your learning algorithm, create a dictionary that maps
## each attribute to all of its possible values. Then reference that
## to find all the possible values.

def make_tree(dataframe) :


## Assume that we are being provided a pandas series named to_classify, and
## we are to return the classification for this data.
## This is also recursive.
## Base case. We are a leaf. Return tree.classification.
## Recursive step. What attribute do we test? Call classify on the child
# corresponding to the value of that attribute in tree.children
def classify(tree, to_classify) :
   pass

