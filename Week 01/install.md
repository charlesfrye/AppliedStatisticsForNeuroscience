# Python Installation Instructions

The homeworks in this course should all be done in Python 3.4 or greater. Jupyter Notebooks will be provided for all assignments. If you do not have Python installed already, we highly recommend using an Anaconda Python installation (described below) to do these assignments. 

The main Python libraries you'll use are

- numpy: library for matrix operations and general mathematical operations
- scipy: a library of modeling tools common in research and data science
- pandas: dataframes for managing complex datasets
- matplotlib: general-purpose plotting library
- seaborn: special plotting library for statistical visualization
- jupyter: "notebook" style documents for python code

### Checking for a previous Python installation
If you already have Python 3.4+ installed, we will create a new virtual environment for NEUR299 which should not interfere with your current Python installation (Case 1).

If you do not have Python 3.4+ installed, we recommend using Anaconda Python and the Anaconda virutal environment (Case 2). Instructions for each case are below, please only follow one, NOT both.

For non-Windows users, to check if you have Python 3, just open up a terminal and type:
```
python3 --version
```
The prompt will either return with the Python version (e.g. Python 3.5.2) or an error (e.g. `python3: command not found`). If it gives you a version that is 3.4 or greater, then you can go directly to setting up a virtual environment. If you do not have Python installed, follow the Anaconda installation instructions. **If you already have a Anaconda Python installed, skip to "Creating a Conda Environment".**

The same instructions should work from the Windows command prompt, possibly with `python` instead of `python3`.

### Setting up a virtual environment - If you already have Python installed [Case 1]
Virtual environments are a way of running multiple Python versions with different libraries without causing conflicts. We will create a workspace for VS265 that has all of the required libraries and will not interfere with other Python work you may have.

First, you'll want to create a folder for this course. In your file explorer, create a folder named ```vs265``` wherever you would like to store the assignments. 

From a terminal, change directories to your ```vs265``` folder. Now we'll setup the virtual environment. In the same terminal run:
```
pip install --upgrade virtualenv
virtualenv -p python3 --system-site-packages neur299env
source neur299env/bin/activate
pip3 install --upgrade numpy scipy matplotlib pandas jupyter seaborn 
```
Now, whenever you want to work on NEUR299 assignments, you can run
```
source /absolute/path/to/neur299env/bin/activate
```
and when you're done and want to return to your regular bash environment you can run
```
deactivate
```
The virtual environment can be activated from any directory. It is simply a way to control which Python version is being used and what libraries are available.

We would recommend that you set up a bash alias to run the ```source /absolute/path/to/neur299env/bin/activate``` command. If you don't know how to do this, ask the GSI.

### Installing Anaconda Python 3 - If you do not already have Python [Case 2]

**Anaconda Python install instructions**

Anaconda is an installation of Python that also manages multiple computing environments. You can download Anaconda from https://www.continuum.io/downloads. Select your operating system and then choose a Python version that is at least 3.4. Follow the instructions on installing for your operating system.

**Creating a conda virtual environment**

We recommend that you set up a conda virtual environment so that you are sure of your environment while working on coursework material. To do this, type:
```
conda create -n neur299env python=3.4 numpy scipy matplotlib pandas jupyter seaborn 
```

This will create a _conda_ virtual environment that has the required libraries. 
Now, whenever you want to work on NEUR299 assignments, you can run
```
source activate neur299env
```
and when you're done and want to return to your regular bash environment you can run
```
source deactivate
```
The virtual environment can be activated from any directory. It is simply a way to control which Python version is being used and what libraries are available.

We would recommend that you set up a bash alias to run the ```source activate neur299env``` command. If you don't know how to do this, ask a GSI!

# Trouble with any of the above?
Come see your friendly GSI!
