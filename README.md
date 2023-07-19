# Neural Implicit 3D Reconstruction

## Using  this code

This code was designed to be model and dataset agnostic. It's aim is to allow for easier and more rapid experimentation with neural implicit reconstruction.
In order to use the code please for your projects please use the following steps:


1. In run_pipeline.py adjust the hyperparameters for you use case.
2. Simply running run_pipeline.py is sufficient.
3. In order to use your own datasets please adjust the input directory to the to raw dataset.
4. In order to preprocess this raw dataset please select from the functions available and add them to the preprocessing parameters in the run_pipeline code.
5. In order to add new models please place the new model architecture in the file Model_Architectures/Model.py. Then make sure to specify the model name in the run_pipiline parameters class.

