FROM andrewosh/binder-base

MAINTAINER Charles Frye <charlesfrye@berkeley.edu>

USER root


RUN pip install searborn==0.7.1
