# Locally weighted regression

Unlike in linear regression where we optimize the weights by taking all training examples in consideration with locally weighted regression we only optimize near a target prediction point xPrime.

We do this by including the weights function in the cost function `C(Q)=W(x)(H(x)-y)^2`

where the `w(x)=exp(-(x-xPrime)^2/2*tau^2)` if tau increase the optimized are increases. The affected area have similar graph to normal distrubution.
