# SqueezeNet-TF2

SqueezeNet is the name of a deep neural network for computer vision that was released in 2016. 
SqueezeNet was developed by researchers at DeepScale, University of California, Berkeley, and 
Stanford University. In designing SqueezeNet, the authors' goal was to create a smaller neural 
network with fewer parameters that can more easily fit into computer memory and can more easily 
be transmitted over a computer network.

# Set up the GPU memory size to avoid the out-of-memory error if the GPU setting has a problem 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

Please remember that it is the TensorFlow realization with image_data_foramt = 'channels_last'. 
If the env of Keras is 'channels_first', please change it according to the TensorFlow convention.  
Make the necessary changes to adapt to TensorFlow 2.2, Keras 2.4.3 and cuDNN 8.0.1. 

In addition, write the new lines of code to replace the deprecated code. The script has many 
modifications on the foundation of the models by Francios Chollet, qubvel and other published 
results. I would like to thank all of them for the contributions.
