# Automatic Differentiation

Automatic differentiation is very important in machine learning, as it handles automatic computation of gradients for gradient descent and other parameter optimization algorithms. In order to better understand how it works, I created this repo to explore different techniques of automatic differentation in python.

## Recursive Differentiation (Forward Accumulation)
The most basic form of automatic differentiation is forward accumulation, which is the first algorithm I implemented. Each operation is represented as a python class which takes in other operations as inputs and contains a `v` variable and `d(var)` lambda function. `v` computes the value of that operation recursively, and `d(var)` computes the gradient of that operation with respect to the input variable recursively. The base case is a `Var` operation which contains a single float value, and a gradient of 1 or 0 depending on whether it is the independent or dependent variable in the gradient. 