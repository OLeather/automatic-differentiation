# Automatic Differentiation

Automatic differentiation is very important in machine learning, as it handles automatic computation of gradients for gradient descent and other parameter optimization algorithms. In order to better understand how it works, I created this repo to explore different techniques of automatic differentation in python.

## Recursive Differentiation (Forward Accumulation)
The most basic form of automatic differentiation is forward accumulation, which is the first algorithm I implemented. Each operation is represented as a python class which takes in other operations as inputs and contains a `v` variable and `d(var)` lambda function. `v` computes the value of that operation recursively, and `d(var)` computes the gradient of that operation with respect to the input variable recursively. The base case is a `Var` operation which contains a single float value, and a gradient of 1 or 0 depending on whether it is the independent or dependent variable in the gradient. 

### Examples
```
- Operator: Var
  - inputs: v
  - v = v
  - d(var) = 1 if (self is var) else 0
- Operator: plus
  - inputs: x, y
  - v = x.v + y.v
  - d(var) = x.d(var) + y.d(var)
- Operator: mul
  - inputs: x, y
  - v = x.v*y.v
  - d(var) = x.v * y.d(var) + y.v * x.d(var)

x = var(5)
y = var(2)

w = 2*x + x
w' = 2 + 1
w = plus(mul(y, x), x)
w.v = plus(mul(y, x), x).v # initial definition
    = mul(y, x).v + x.v # recursively expand
    = (y.v * x.v) + x.v # recursively expand
    = (2 * 5) + 2 
    = 12 

w.d(x) = plus(mul(y, x), x).d # initial definition
       = mul(y, x).d(x) + x.d(x) # recursive expand
       = (y.v * x.d(x) + x.v * y.d(x)) + x.d(x) # recursive expand
       = (2 * 1 + 5 * 0) + 1 
       = 7
```
