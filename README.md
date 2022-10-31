# math-898
Repo to organize work related to SFSU thesis on NN verification.

# Install (currently only tested on MAC OSX command line)
```
# download this repo via command line
git clone git@github.com:roarkm/math-898.git

# navigate into directory
cd math-898/python

# create a python virtual environment to install packages locally into .venv dir
python3 -m venv .venv

# switch to the created virtual environment
source .venv/bin/activate
python3 -m pip install --upgrade 'pip==22.3'
# not sure why, but this has to be installed before requirements
pip install numpy numpy==1.23.2

# install required packages to virtual environment
pip install -r requirements.txt

# run a script
python src/models/multi_layer_sdp.py

# exit the virtual environment or just close your shell window
deactivate
```
