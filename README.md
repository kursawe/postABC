# postABC
This is a python project for ABC rejection and regression. In order to use the postABC functionality it is necessary to generate samples of the model according to the prior distribution first. ABC rejection and regression can then be conducted by passing the model output to the project as tables in the form of numpy arrays. The usage is explained in the docstrings in ./src/postABC.py and examples are provided in the ./test folder.

Current functionality includes:

* multivariate ABC rejection and regression as introduced by Beaumont (2002)
* least-squares multivariate weighted kernel density estimation using the epanechnikov kernel
