# SqueezeNet-TF2

## Introduction 

SqueezeNet is the name of a deep neural network for computer vision that was released in 2016. 
SqueezeNet was developed by researchers at DeepScale, University of California, Berkeley, and 
Stanford University. In designing SqueezeNet, the authors' goal was to create a smaller neural 
network with fewer parameters that can more easily fit into computer memory and can more easily 
be transmitted over a computer network.

## Modifications 

Please remember that it is the TensorFlow realization with image_data_foramt = 'channels_last'. 
If the env of Keras is 'channels_first', please change it according to the TensorFlow convention.
Make the necessary changes to adapt to TensorFlow 2.2, Keras 2.4.3 and cuDNN 8.0.1. 

In addition, write the new lines of code to replace the deprecated code. The script has many 
modifications on the foundation of the models by Francios Chollet, qubvel and other published 
results. I would like to thank all of them for the contributions.
