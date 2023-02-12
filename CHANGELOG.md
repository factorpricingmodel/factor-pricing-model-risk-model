# Changelog

All noteable changes to this project will be documented in this file.

<!--next-version-placeholder-->

## v2023.1.0 (2023-02-12)
### Feature
* Introduce cov half life ([`92d96dc`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/92d96dc5449e01dca49f0dce2d159891e082022e))
* Add cov halflife into the accuracy functions ([`cd20be4`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/cd20be4bb91bd61435cfc1ff1791111f36424ce2))
* Support exponential weighted factor covariance ([`ffac512`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/ffac512c0012aa372cc8ed7fed98f48df2c1e747))
* Support exponential weighted least square ([`db4ae53`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/db4ae5360892c4d941fe3f815cfd0a26269012b4))

### Fix
* Ewm in cov ([`f34ddbf`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/f34ddbfeb4c2409bed1f3ecddda2f1bdccd6bb60))

### Documentation
* Fix example config ([`149a660`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/149a6607b008af0a7e19569e2302b4e5318f11f3))
* Fix grammar ([`cb4a456`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/cb4a45625d2bb26b552fadead498f3ebe3059dd9))

## v2023.0.0 (2023-01-12)
### Feature
* Add value at risk accuracy ([`7034a8e`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/7034a8e2455c1709d7ef63ae71d98cf50c272cb7))

### Fix
* Another bug created from the previous bugfix ([`d48381a`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/d48381a1fb6e414e125b45bceae648b14823a3b5))
* Bug in verifying ndarry existence ([`f42bd2d`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/f42bd2d38d9c6ba7f1680143fc5fb33182917262))
* Typo in parameter ([`f5c5dab`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/f5c5dabbd0c07668372766d63215e316543ed880))
* Correct the forecast return computation ([`6603ce0`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/6603ce00eb5e955c7203905c92496f395e030250))

### Documentation
* Add rolling risk model description ([`e20bb28`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/e20bb28c1661184cdf826b013b1ce41fbd3f4a3c))
* Update README ([`a8fd716`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/a8fd71669bd2e0c35b1b2284404663639d7c4e73))
* Correct typo ([`d15e684`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/d15e684f7a9bbbf00368cf0c75a35ab848b1050a))
* Update bias and VaR documentation page ([`e9c7542`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/e9c7542da8d3abfbbf433e85b42537de0d622250))
* Add bias and var placeholder pages ([`55d79b4`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/55d79b43f768a61201bad7451326a356d1edd780))
* Correct factor risk model title ([`239c41f`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/239c41f5d2ac16ff3c1a18b2a123162776c4e77c))
* Add factor risk model doc ([`b35ca51`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/b35ca518d833bcc939112b18b00b5724b96ee5a4))
* Add risk model descrption ([`40af5a6`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/40af5a6006e513c6ed0934c8ea22a5871e8b1301))
* Update README ([`5877781`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/5877781d8c64e99155da596fdc5a31b0c8d90e51))
* Correct caller name ([`62c7df9`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/62c7df9c025a7d384124b21c13170a48b042814c))

## v2022.1.0 (2022-12-30)
### Feature
* Add equal weight portfolio pipeline ([`a2a7db6`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/a2a7db6a01c92e7819d0c5d5ee4800bd250913f9))
* Add bias statistics ([`efe3f46`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/efe3f46f4305ce77833e445cf7529558c5e6b2d5))
* Add standardized returns in bias check ([`128c3f2`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/128c3f20383e6d62e91f3ad30fc11b3adc23b7d5))
* Get covariance and correlation from risk model ([`02b606d`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/02b606dc9f95320e83943a019c1a35ab315cb6b7))
* Support transforming rolling factor risk model ([`77810b0`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/77810b0867ff37e2a64c02f9acc9f07222d804d3))
* Support exporting rolling factor risk model ([`5da7006`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/5da7006de46fa278e086881743fc8adc5822f479))
* Support factor risk model transformer ([`e26b61c`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/e26b61c59b8515ab4853c2adecd16d60b8177eee))
* Support factor risk model transformer ([`770ba6a`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/770ba6a619e937c78a0f308a33990dd509e5d169))
* Add factor covariance attribute in risk model ([`0b15a40`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/0b15a401897e6f5e87a8300aed85fc59e6a15709))
* Add WLS regressors ([`2b8d60b`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/2b8d60b8ba00a719d1a8d5d2721719636a3f3f1d))
* Change risk model input format to indexed on date / time rather than instruments ([`267d9a4`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/267d9a4e1809e3709c57d67f1661655353314ee6))
* Add parameter to show progress in rolling PCA ([`35fc9e3`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/35fc9e3130805ebf1b2cb9fd955e17187938a3b1))
* Add statistical risk model Rolling PCA ([`603344c`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/603344cafcaed8179d075e8fcfe2e9975a931c29))
* Add the first statistical model PCA ([`4da36f7`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/4da36f7987e8876bc615b721aa41d1b9462f003c))

### Fix
* Incorrect mapping of factor exposures ([`4bc0028`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/4bc00285f5dbb38e644d67c0c245c988e5e614e0))

### Documentation
* Add accuracy example configuration ([`f8b104a`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/f8b104a26028f66947569c5fc2ef54fd24f8e7b0))
* Correct typo ([`46d3202`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/46d32028c22ff58349c478927c9abbe6145a984c))
* Update PCA documentation ([`9523197`](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/commit/952319735778302aae950a0e1160b659960650ae))

## [Unreleased]
