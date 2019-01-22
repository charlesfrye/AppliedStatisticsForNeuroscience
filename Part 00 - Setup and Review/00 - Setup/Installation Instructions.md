# Python Installation Instructions

This course was written in Python 3.6.
While it may be possible to run in later versions of Python,
this is not guaranteed.
If you do not have Python installed already, we recommend using an Anaconda Python installation (described below).

The main Python libraries you'll use are

- numpy: library for matrix operations and general mathematical operations
- scipy: a library of modeling tools common in research and data science
- pandas: dataframes for managing complex datasets
- matplotlib: general-purpose plotting library
- seaborn: special plotting library for statistical visualization
- jupyter: "notebook" style documents for python code

### Installing Anaconda Python 3

**Anaconda Python install instructions**

Anaconda is an installation of Python that also manages multiple computing environments. You can download Anaconda from https://www.continuum.io/downloads. Select your operating system and then choose a Python version that is at least 3.6. Follow the instructions on installing for your operating system. **NOTE FOR WINDOWS USERS**: while installing, make sure you check the box that says  "Add to $PATH". This may cause the text to turn red.

**Creating a ```conda``` virtual environment**

We recommend that you set up a conda virtual environment so that you are sure of your environment while working on coursework material. To do this, you'll need the environment specification file `environment.yml` from the GitHub repo.
Once you've acquired it and placed it in the appropriate folder (this is automatic if you clone or download the repo),
run the following command in the command line:
```
conda create --file environment.yml -n neur299
```

This will create a `conda` virtual environment that has the required libraries. 
Now, whenever you want to work on NEUR299, you can run
```
source activate neur299
```
to "enter" the new environment, and when you're done and want to return to your regular environment you can run
```
source deactivate
```
The virtual environment can be entered from any directory. It is simply a way to control which Python version is being used and what libraries are available.

**NOTE FOR WINDOWS USERS**: your commands will be `activate neur299` and `deactivate`, respectively.

**Linking ```conda``` and Jupyter**

We'll be using a set of tools from the
[Jupyter Project](http://jupyter.org/)
for this course.
In order to integrate Jupyter into our new `conda` environment,
we need to run the following command _while in that environment_:

```
python -m ipykernel install --user --name neur299 --display-name "neur299"
```
