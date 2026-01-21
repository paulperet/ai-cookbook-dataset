# AutoRec: A Neural Collaborative Filtering Model for Rating Prediction

## Overview

This guide demonstrates how to implement AutoRec, a neural network-based collaborative filtering model that uses an autoencoder architecture to predict user ratings. Unlike traditional matrix factorization, AutoRec can capture complex nonlinear relationships in user-item interactions.

### Prerequisites

Ensure you have the required libraries installed. This implementation uses MXNet.

```bash
pip install mxnet d2l
```

## 1. Import Libraries

We begin by importing the necessary modules.

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## 2. Define the AutoRec Model

The AutoRec model consists of an encoder and a decoder. The encoder transforms the input into a hidden representation, and the decoder reconstructs the original input. We apply dropout after the encoder to prevent overfitting.

**Key Design:** During training, we mask the gradient so that only observed ratings (non-zero entries) contribute to the learning process.

```python
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid', use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## 3. Implement the Evaluation Function

We need a custom evaluator because AutoRec operates on the entire interaction matrix column (item-based). We use Root Mean Square Error (RMSE) as our accuracy metric, considering only the observed ratings in the test set.

```python
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## 4. Prepare the Data

We'll use the MovieLens 100K dataset. The data is split into training and test sets, and then converted into interaction matrices suitable for the item-based AutoRec model.

```python
# Load and split the data
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)

# Create interaction matrices
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users, num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users, num_items)

# Create data loaders
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
```

## 5. Train and Evaluate the Model

Now, we initialize the model, define the training parameters, and start the training loop. We'll use the Adam optimizer and L2 loss.

```python
# Set up devices (GPUs if available)
devices = d2l.try_all_gpus()

# Initialize the model
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))

# Define training hyperparameters
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

# Train the model
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

After training, the model will output the test RMSE. You should observe that AutoRec achieves a lower RMSE than traditional matrix factorization, demonstrating the advantage of using neural networks for this task.

## Summary

* AutoRec frames collaborative filtering as an autoencoder problem, enabling the capture of nonlinear user-item relationships.
* The model uses gradient masking to ensure only observed ratings influence training.
* On the MovieLens 100K dataset, AutoRec outperforms standard matrix factorization.

## Exercises

1. **Vary the hidden dimension:** Experiment with different values for `num_hidden` (e.g., 200, 500, 1000) to see its impact on model performance and training time.
2. **Add more hidden layers:** Modify the `AutoRec` class to include additional hidden layers. Does this improve the RMSE?
3. **Experiment with activations:** Try different activation functions (e.g., `relu`, `tanh`) for the encoder and decoder. Can you find a combination that yields better results?

---
*For further discussion, visit the [D2L forum](https://discuss.d2l.ai/t/401).*