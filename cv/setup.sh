# Install Docker
# TODO
# Add docker user
sudo groupadd docker
sudo usermod -aG docker $USER
# Log into new group:
newgrp docker

# Train Tensorflow MNIST recognition
# This model is used to check that the draw model
# is functioning correctly.
./dockerrun 'python ./src/draw/mnist.py'

# Train Tensorflow Draw
./dockerrun 'python ./src/draw/tf_draw.py'
