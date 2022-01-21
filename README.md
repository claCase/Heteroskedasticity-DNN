# Heteroskedasticity Regression
Full Log-Likelihood Heteroskedastic Regression with Deep Neural Networks and Tensorflow
# Linear <br/>
### Deterministic parameters with no epistemic uncertainty 
```console
--epochs 50 --droprate 0.2 --reduction 7 --mean "linear"
```
![alt-text](https://github.com/claCase/Heteroskedasticity-DNN/blob/main/Figures/sampleslinear_deterministic.png)
![alt-text](https://github.com/claCase/Heteroskedasticity-DNN/blob/main/Figures/paramslinear_deterministic.png)
### Stochastic parameters with epistemic uncertainty 
```console
--epochs 50 --epistemic --droprate 0.2 --reduction 7 --mean "linear"
```
![alt-text](https://github.com/claCase/Heteroskedasticity-DNN/blob/main/Figures/sampleslinear_epistemic.png)
![alt-text](https://github.com/claCase/Heteroskedasticity-DNN/blob/main/Figures/uncertaintylinear_epistemic.png)

# Non Linear <br/>
### Deterministic parameters with no epistemic uncertainty  
```console
--epochs 50 --droprate 0.2 --reduction 7 --mean "cosine"
```
![alt-text](https://github.com/claCase/Heteroskedasticity-DNN/blob/main/Figures/samplescosine_deterministic.png)
![alt-text](https://github.com/claCase/Heteroskedasticity-DNN/blob/main/Figures/paramscosine_deterministic.png)
### Stochastic parameters with epistemic uncertainty  
```console
--epochs 50 --epistemic --droprate 0.2 --reduction 7 --mean "cosine"
```
![alt-text](https://github.com/claCase/Heteroskedasticity-DNN/blob/main/Figures/samplescosine_epistemic.png)
![alt-text](https://github.com/claCase/Heteroskedasticity-DNN/blob/main/Figures/uncertaintycosine_epistemic.png)
