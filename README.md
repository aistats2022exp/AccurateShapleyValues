# Accurate Shapley Values for tree-based model
 
ACV is a python library that provides **a better estimation of 
Shapley values (SV) for tree-based model** (>= dependent TreeSHAP). 
In addition, we use the coalition version of SV to properly handle categorical variables in the computation of SV.


## Requirements
Python 3.6+ 

**OSX**: ACV uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)

**Windows**: Install MinGW (a Windows distribution of gcc) or Microsoft’s Visual C

## Installation

Clone the repo and run the following command in the main directory
```
$ python setup.py install
```

## How does ACV work?
ACV works for XGBoost, LightGBM, CatBoostClassifier, scikit-learn and pyspark tree models. To use it, we need to transform our model into ACVTree.
### Example:

```python
from acv_explainers import ACVTree

tree_based = RandomForestClassifier() # or any tree-based models
#...trained the model

# Initialize the explainer
acvtree = ACVTree(tree_based, data) # data should be np.ndarray with dtype=double
```
### Shapley Values of categorical variables
Let assume we have a categorical variable Y with k modalities that we encoded by introducing the dummy variables <img src="https://latex.codecogs.com/gif.latex?Y_1%2C%5Cdots%2C%20Y_%7Bk-1%7D" />. As show in the paper, we must take the coalition of the dummy variables to correctly calculate the Shapley values.

```python

# cat_index := list(list) that contains the index of the dummies or one-hot variables grouped 
# together for each variable. For example, if we have only 2 categorical variables Y, Z 
# transformed into [Y_0, Y_1, Y_2] and [Z_0, Z_1, Z_2]

cat_index = [[0, 1, 2], [3, 4, 5]]
forest_sv = acvtree.shap_values(X, C=cat_index)
```
In addition, we can compute the SV given any coalitions. For example, if we want the following coalition <img src="https://latex.codecogs.com/gif.latex?C_0%20%3D%20%28X_0%2C%20X_1%2C%20X_2%29%2C%20C_1%3D%28X_3%2C%20X_4%29%2C%20C_2%3D%28X_5%2C%20X_6%29" />

```python

coalition = [[0, 1, 2], [3, 4], [5, 6]]
forest_sv = acvtree.shap_values(X, C=coalition)
```
* **Finally, to compute the classic SV that is without coalition. We set C=[[]]**

```python
# Classic SV w.o coalition 
coalition = [[]]
forest_sv = acvtree.shap_values(X, C=coalition)
```

You will find the experiments of the paper: [HERE](https://github.com/aistats2022exp/AccurateShapleyValues/tree/main/notebook)