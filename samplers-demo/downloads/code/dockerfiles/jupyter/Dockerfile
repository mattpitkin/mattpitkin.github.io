FROM jupyter/minimal-notebook
# Jupyter minimal notebook installs Ubuntu 16.04 with a jupyter notebook with a python3 kernel

WORKDIR /samplers

USER root

### Distribution package installs
RUN apt-get update

# install git and gfortran, pkg-config, BLAS, cmake, libmpich-dev
RUN apt-get install -y git gfortran pkg-config libblas-dev liblapack-dev libmpich-dev cmake libopenmpi-dev scons dvipng

# Install JAGS
RUN apt-get install -y jags

# clean things up
RUN apt-get -y autoremove \
    && apt-get -y clean

### Python modules installed via conda and pip

# Install numpy, scipy
USER $NB_USER
RUN conda install -y numpy
RUN conda install -y scipy
USER root

# Install matplotlib and corner
RUN pip install --trusted-host pypi.python.org matplotlib corner

# Install samplers
# Install PyJAGS from PyPI
RUN pip install --trusted-host pypi.python.org pyjags

# Install emcee
RUN pip install --trusted-host pypi.python.org emcee

# Install PyStan
RUN pip install --trusted-host pypi.python.org pystan

# Install PyMC3
RUN pip install --trusted-host pypi.python.org pymc3

# Install Nestle
RUN pip install --trusted-host pypi.python.org nestle

# Install CPNest
RUN pip install --trusted-host pypi.python.org cpnest

# Install MultiNest 
ADD MultiNest_v3.10_CMake.tar.gz /samplers
RUN cd MultiNest_v3.10_CMake/multinest/build \
    && cmake .. && make && make install && cd

# Install PyMultiNest
RUN pip install --trusted-host pypi.python.org pymultinest

ENV LD_LIBRARY_PATH /usr/local/lib

# Install DNest4 (and clean up so that Python2 install can also proceed)
RUN cd ${HOME} && git clone https://github.com/eggplantbren/DNest4.git
RUN chown -R $NB_USER ${HOME}/DNest4
RUN cd ${HOME}/DNest4/python && python setup.py install --prefix=/opt/conda \
    && rm -rf build && rm -rf dist && rm -rf dnest4.egg-info && rm dnest4/_dnest4.cpp

# set working directory as "/home/jovyan/work"
WORKDIR ${HOME}/work

# Add PolyChord (don't install for Python 3 as it does not work yet)
ADD PolyChord_v1.12.tar.gz ${HOME}

# Install PyPolyChord (don't use the make file, use setup.py)
RUN cd ${HOME}/PolyChord && make MPI= && touch PyPolyChord/.ld_preload.sh; ldd lib/libchord.so | grep -o '/.*libmpi.so[^/]* ' | awk '{print "export LD_PRELOAD="$$1":$$LD_PRELOAD"}' > PyPolyChord/.ld_preload.sh; python setup.py install && chmod -R a+rw ${HOME}/PolyChord

# Install getdist for PolyChord post-processing
RUN pip install --trusted-host pypi.python.org getdist

# switch back to non-root user
USER $NB_USER

# add Python 2 kernel for Jupyter notebook (PyPolyChord needs Python 2 at the moment, disable MPI for PolyChord)
RUN conda create -y -n py27 python=2.7 && /bin/bash -c "source activate py27; conda install -y notebook ipykernel; conda install -y numpy; conda install -y scipy; pip install matplotlib corner; pip install emcee pymc3 pystan pyjags pymultinest nestle cpnest getdist; cd ${HOME}/DNest4/python; python setup.py install --prefix=/opt/conda/envs/py27; cd ${HOME}/PolyChord; python setup.py install --prefix=/opt/conda/envs/py27; ipython kernel install --user"

USER root
# add global environment variables
RUN echo "LD_LIBRARY_PATH=/usr/local/lib:${HOME}/PolyChord/lib" >> /etc/environment

# switch back to non-root user
USER $NB_USER
