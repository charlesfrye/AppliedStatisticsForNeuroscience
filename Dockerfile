FROM andrewosh/binder-base

MAINTAINER Charles Frye <charlesfrye@berkeley.edu>

USER root


RUN pip install seaborn==0.7.1
