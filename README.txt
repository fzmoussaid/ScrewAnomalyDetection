This project has two parts:

1/ A folder that contains the source code that launches the training process on different set of hyperparameters.
The objectif is find the best combination of hyperparameters.
To launch the process, just run:
python main.py
However, you will need to install tensorflow.
You can then visualize the results using tensorboard. You can do it by running the following command:
tensorboard --logdir logs/hyperparam-tuning
2/ A Jupyter notebook that shows the results of the binary classification using a model with 
the best hyperparameters.