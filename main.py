import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
from tensorflow_probability import distributions as tfd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.animation as anim

if __name__ == "__main__":
    save_dir = os.path.join(os.getcwd(), "Figures")
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--name", type=str, default="1")
    parser.add_argument("--mean", type=str, default="linear")
    parser.add_argument("--reduction", type=int, default=10)
    parser.add_argument("--epistemic", action="store_true")
    parser.add_argument("--droprate", type=float, default=0.5)
    args = parser.parse_args()
    save = args.save
    epochs = args.epochs
    name = args.name
    mean_type = args.mean
    red = args.reduction
    epistemic = args.epistemic
    drop_rate = args.droprate
    col_pred = "darkviolet"
    col_true = "dodgerblue"
    samples = 10000
    batches = 2
    x = np.linspace(0, 2 * np.pi, samples)
    sigma = np.cos(x) * 5 + 6
    if mean_type == "linear":
        mu = x * 5
    elif mean_type == "cosine":
        mu = np.cos(x * 3) * 5
    target_data = np.empty(shape=(batches, samples))
    input_data = np.empty(shape=(batches, samples))
    for i in range(batches):
        normal_samples = np.random.normal(loc=mu, scale=sigma, size=samples)
        target_data[i] = normal_samples
        input_data[i] = x

    input_model_data = input_data.flatten().reshape(-1, 1)
    target_model_data = target_data.flatten().reshape(-1, 1)
    plt.figure(figsize=(10,7))
    plt.scatter(input_model_data.flatten()[::red], target_model_data.flatten()[::red], color=col_true, label="True Samples", s=3)

    # Create Neural Network model for mean and variance
    i = l.Input(shape=(1,))
    f = l.Dense(100, "relu")(i)
    if epistemic:
        f = l.Dropout(drop_rate)(f)
    f = l.Dense(100, "relu")(f)
    if epistemic:
        f = l.Dropout(drop_rate)(f)
    f = l.Dense(100, "relu")(f)
    if epistemic:
        f = l.Dropout(drop_rate)(f)
    mean_var = l.Dense(2, None)(f)
    model = m.Model(i, mean_var)

    # Create Normal distribution for log-likelihood loss
    def variance(v):
        return 1e-3 + tf.math.softplus(0.05 * v)

    def distr(x):
        return tfd.Normal(loc=x[..., :1], scale=variance(x[..., -1:]))

    # Compile model for optimization
    model.compile(optimizer="adam", loss=lambda y, y_hat: -distr(y_hat).log_prob(y))
    # Optimize the model with negative log-likelihood loss
    history = model.fit(x=input_model_data, y=target_model_data, epochs=epochs, shuffle=True, validation_split=.3)
    loss = history.history

    # Predict mean and variance logits
    preds = model.predict(input_model_data)
    # Samples from the model distribution
    model_samples = distr(preds).sample(1)

    # Plot
    plt.suptitle("Aleatoric Uncertainty")
    plt.scatter(input_model_data.flatten()[::red], model_samples.numpy().flatten()[::red], color=col_pred,
                label="Model Samples", s=2)
    plt.title("Samples")
    plt.legend()
    if save:
        plt.savefig(os.path.join(save_dir, f"samples_{name}.png"))

    # Plot preds
    preds2 = model.predict(x[:, None])
    mu_pred = preds2[:, 0]
    sigma_pred = variance(preds2[..., -1:])
    sigma_pred = sigma_pred[:]
    fig, ax = plt.subplots(2, figsize=(10,7))
    fig.suptitle("Parameters Prediction")
    ax[0].plot(x, mu_pred, label="Pred Mean", color=col_pred)
    ax[0].plot(x, mu, label="True Mean", color=col_true)
    ax[0].set_title("mu")
    ax[0].legend()
    ax[1].plot(x, sigma_pred, label="Pred Variance", color=col_pred)
    ax[1].plot(x, sigma, label="True Variance", color=col_true)
    ax[1].set_title("Sigma")
    ax[1].legend()
    plt.tight_layout()
    if save:
        plt.savefig(fname=os.path.join(save_dir, f"params_{name}.png"))

    if epistemic:
        # Plot Epistemic Uncertainty
        parameters_samples = []
        for i in range(10):
            params = model(x[::red, None], training=True)
            parameters_samples.append(params)

        fig3, ax3 = plt.subplots(2, figsize=(10,7))
        fig3.suptitle("Data and Model Samples: Epistemic Uncertainty")
        for i in range(len(parameters_samples)):
            params = parameters_samples[i]
            mu_pred = params[:, 0]
            mu_pred = mu_pred.numpy().flatten()
            var_pred = variance(params[:, 1:])
            var_pred = var_pred.numpy().flatten()
            if i == len(parameters_samples)-1:
                ax3[0].scatter(x[::red], mu_pred, color=col_pred, s=1, label="Pred Mean samples")
                ax3[1].scatter(x[::red], var_pred, color=col_pred, s=1, label="Pred Variance samples")
            else:
                ax3[0].scatter(x[::red], mu_pred, color=col_pred, s=1)
                ax3[1].scatter(x[::red], var_pred, color=col_pred, s=1)
        ax3[0].plot(x, mu, label="True Mean", color=col_true, linewidth=3)
        ax3[1].plot(x, sigma, label="True Variance", color=col_true, linewidth=3)
        ax3[1].set_title("Sigma")
        ax3[0].set_title("mu")
        plt.legend()
        if save:
            plt.savefig(fname=os.path.join(save_dir, f"uncertainty_{name}.png"))

    plt.figure(figsize=(10,7))
    labels = ["Train Log-Likelihood", "Test Log-Likelihood"]
    for i, k in enumerate(loss.keys()):
        plt.plot(loss[k], label=f"{labels[i]}")
    plt.legend()
    if save:
        plt.savefig(fname=os.path.join(save_dir, f"loss_{name}.png"))

    plt.show()
