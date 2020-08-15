NumPy implementation of the DRAW network.

# Code
Most interesting files are:

    src/draw/vanilla_draw.py
    test/test_tf_draw.py

# Setup
Requirements:
  * Linux system with an Nvidia graphics card.
  * Nvidia drivers installed (CUDA libraries are *not* necessary).
  * Docker

# Run
Create the Docker image and start a container then running a given command.

    ./dockerrun <command>

## Examples
Drop into a bash shell of the container:

    ./dockerrun bash

Run all of the tests:

    ./dockerrun pytest

Train the MNIST classifier:

    ./dockerrun ./src/draw/mnist.py

Train the Tensorflow DRAW implementation:

    ./dockerrun ./src/draw/tf_draw.py

Train and test the NumPy DRAW implementation:

    ./dockerrun 'pytest -k test_model'

Note: the MNIST classifier must be trained before the Tensorflow or NumPy
DRAW implementation tests can be run, as they require a functioning classifier.

# Possible Docker issues
## Run docker without sudo 
To avoid running the Docker commands as root, create a docker user:

    # Add docker user
    sudo groupadd docker
    sudo usermod -aG docker $USER
    # Log into new group:
    newgrp docker

## Root owned folders created by Docker
I have had some issues with Docker creating the ./out and ./data directories as
root owned directories. If this occurs, a recursive chown is needed so that 
Docker can be run without root privilages. 


