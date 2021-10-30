FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget vim nano git

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.8.3-Linux-x86_64.sh

RUN conda --version

RUN conda install pytorch=1.4.0 torchvision=0.2.1 cudatoolkit=10.1 -c pytorch
RUN conda install pandas scikit-learn matplotlib
RUN conda install -c conda-forge notebook
