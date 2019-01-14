# Decision Tree Classifier


This project aims to show DTC algorithm decision boundaries on simple data sets.

### Project description

---

Decision Tree Classifier tends to overfit data and can be hard to visualize when multiple dimensions are taken into consideration. To overcome this issue this project finds only two best features of a data set using DTC and then fits the predictor on those two features on new DTC instance to find decision boundaries. As the purpose of this project is to simply help to grasp how DTC works, its max depth (number of 'branch levels') is limited to 2.

First of all, it is handy to present algorithm's results on a graph.

![Imgur](https://i.imgur.com/12eKj43.png)

The algorithm divided the data on value of 2.45 for feature X[1] and on value of 1.75 for feature X[0].  The decision boundaries on a 2D plane look as on the below illustration. Note that each color represents one of three categories classified by the algorithm basing on only two features (in this case petal width and petal length). Decision boundaries are highlighted by red dashed line.

![Imgur](https://i.imgur.com/gH3Olns.png)

The algorithm was run on famous Iris data set, where three categories are related to their specie:
* virginica
* versicolor
* setosa.

### Getting started

---

1. Clone this repo
2. Run bash script `run_all.sh`
