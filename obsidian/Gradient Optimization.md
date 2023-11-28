
# Gradient Optimization method

We can optimize our model parameters using gradient optimization methods either descent or ascent depending on the problem.

What we are doing is taking small steps down or up hill by calculating partial derivative of the cost function to respect of a parameter Qy.

`Qy=Qy +/- alpha*dCost/dQy`

- **alpha** is the learning rate 
- **dcost/dQy** is the partial derivative of the cost function to respect of a parameter Qy.
- `Qy` the parameter.

When training we mainly apply many iterations and we update either with each training example or each batch.

### Batch gradient descent 

In batch gradient descent we update the parameter after passing the whole batch it goes like this :

``` python
def batch_gradient_descent(

self, learning_rate, inputSet, labels, loss_threshold, max_iterations

):

previous_loss = self.one_batch_update(inputSet, labels, learning_rate)

converage = False

iteration = 0

while not converage:

current_loss = self.one_batch_update(inputSet, labels, learning_rate)

if (

abs(current_loss - previous_loss) < loss_threshold

or iteration == max_iterations

):

converage = True

iteration += 1

print(previous_loss)

previous_loss = current_loss
```

## Stochastic gradient descent 

In stochastic gradient descent we update the parameter on each training example it goes as follows:

```python 
def stochastic_gradient_descent(

self, learning_rate, inputSet, labels, loss_threshold, max_iterations

):

previous_loss = self.one_stochastic_update(inputSet, labels, learning_rate)

converage = False

iteration = 0

while not converage:

current_loss = self.one_stochastic_update(inputSet, labels, learning_rate)

if (

abs(current_loss - previous_loss) < loss_threshold

or iteration == max_iterations

):

converage = True

iteration += 1

print(previous_loss)

previous_loss = current_loss
```

###  Notes

- we can use `mini-batch` instead the whole dataset.
- The convergence point is when gradient isnt updating that much we use a threshold like 0.0001.
- some times we use gradient ascent if we want to increase the cost function 