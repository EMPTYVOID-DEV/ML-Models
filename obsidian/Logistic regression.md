# Logistic regression

We can use it for binary classification and we can use one-vs-rest to do multi-class-classification. `Raw=Q0+Q1*X1+...Qn*Xn` , `H(x)=Sigmoid(Raw)` and `Sigmoid=1/1+e^-x` 

so the new hypothesis output ranges from 0 to 1 , to put it more clearly h(x) is calculating the probability of the output been 1.

## The cost function

We use categorical cross-entropy which is defined with this formula :

`L(yPrime=H(x),y)=-(ylog(yPrime)+(1-y)log(1-yPrime))` and since y is either 0 or 1 we have if y=0 => `L(yPrime,y)=-log(1-Yprime)` else  => `L(yPrime,y)=-log(Yprime)`.

## Optimization

we can use batch gradient ascent to maximize the cost function where we optimize each parameter Qj like this :

`Qji+1=Qji + alpha*(sum((yi-H(x)i)xji) from i=1 to m)`

## Multi-class-classification

we can use a method called **one verse rest** to do multi class classification. Here are the setups :

1. create as much models as classes
2. train each model to predict a specific class => outputs 1 when we match that class zero otherwise.
3. to make classify an input we use at most n-1 model where n is number of classes.

