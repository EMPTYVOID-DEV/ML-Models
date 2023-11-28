
Its general linear model with a hypothesis function  that goes as follows :

`H(x)=Q0+Q1*X1+...Qn*Xn` where X1....Xn are the features and Q0....Qn are the parameters.

## The cost function

We use square residual function as a cost function `(y-H(x))^2` in case we use batch gradient descent it gonna be `sum((yi-H(xi))^2)/2 for i from 1 to m`

### Optimization

There is handful of algorithemes we can utilize in order to optimize the parameters **gradient_descent** ,**newtons method** ...etc

We gonna use [Stochastic Gradient descent](obsidian://open?vault=obsidian&file=Gradient%20Optimization) where update each parameter with this formula :

`Qji+1=Qji - alpha*(H(x)i-yi)xji` i :1-m m is number of training examples.

- Qji is the i-th value of Qj where j is the index of the parameter.