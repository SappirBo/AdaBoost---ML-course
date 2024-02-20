# AdaBoost_ML_course

## Details
Implement of the AdaBoost algorithm as part of machine learning course. AdaBoost, short for Adaptive Boosting, is a machine learning algorithm formulated as an ensemble technique that combines multiple weak learners to create a strong learner.<br/>


## Dataset
In our AdaBoost implementation, we are developing a machine learning model to classify data points based on their features. in ```circle_separator.txt``` file, we have data points where each point have coordinates (x, y) and a label (-1 or 1).
<br/><br/>
![image](https://github.com/SappirBo/AdaBoost_ML_course/assets/92790326/2cd86ddd-6ed6-4302-810a-c289a5803f94)


## Hypotheses (rules)
The AdaBoost algorithm in our implementation can use two types of rules:
1. Lines: This rule type uses straight lines to separate data points. A line is defined by two points in the dataset, and the classification of a point depends on which side of the line it falls.
2. Circles: In this rule type, circles are used to classify points. A circle is defined by two points: one point being its center and the other giving the radius. 

![image](https://github.com/SappirBo/AdaBoost_ML_course/assets/92790326/33b1a122-6c5b-40be-9f88-3bb8a1ea4d65)

## Output
The output of the program is 3 parts: 
1 .sum of the operation that activated (number of runs, iterations and hypotheses).
2. Average errors upon all the runs.
3. summary.

![image](https://github.com/SappirBo/AdaBoost_ML_course/assets/92790326/95ac44de-9e85-4a11-aee3-2ceb24424ee1)


## How to run  
first make sure the [installation](https://github.com/SappirBo/AdaBoost_ML_course/blob/main/README.md#installation) works properly.<br/>
<br/>
In ```main.py``` file, you have will have all at once configurations:

![image](https://github.com/SappirBo/AdaBoost_ML_course/assets/92790326/3e893eb3-64cb-4a86-9c5c-6aed4798b533)

you can update the configuration each time for different test cases. 
 
## Installation
First download all the needed libareries, make sure that you have the right Python environment.<br/>

Needed libareries: 
* [NumPy](https://numpy.org/).
* [Matplotlib](https://matplotlib.org/).
To downlad them both, run ```pip install numpy matplotlib```.<br/>
