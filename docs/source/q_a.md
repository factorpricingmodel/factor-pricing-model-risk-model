# Q&A

The following are general questions which provide you overview
of the usage and limitation of the library.

## Is the risk model annualised?

The granularity of the risk model depends on the instrument
returns fitting the risk model. If you input daily returns
into the method `fit`, the covariance matrix returned by the
same risk model contains the pairwise daily covariance entries,
while the correlation is unaffected. In short, you need to manually
annualise your risk model regarding the granularity of fitting
data.

## What are the expected input types?

Currently only Pandas (DataFrame / Series) and NumPy (array) are 
accepted in modeling input, while we are extending to accept Polars
(DataFrame / Series) in the future.