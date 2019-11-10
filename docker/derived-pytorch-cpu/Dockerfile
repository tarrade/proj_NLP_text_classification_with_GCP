# Specify the base image and tag
FROM gcr.io/deeplearning-platform-release/pytorch-cpu:m34

# Checking the type of machine
RUN cat /etc/os-release
RUN lsb_release -a
RUN uname -r

# Checking that the right shell is used in the image the build and later when the container is launched from the image
RUN echo $0
# active in the shell
SHELL ["/bin/bash", "-c"]
# active in the terminla later
ENV SHELL /bin/bash
RUN echo $0

## Auto Shutdown On Idle
#RUN apt-get update && apt-get install -y sysstat bc
#RUN git clone --branch auto_shutdown https://github.com/b0noI/ai-platform-samples.git
#RUN cd ai-platform-samples/notebooks/auto-shutdown && cp ./ashutdown /usr/local/bin/ && cp ./ashutdown.service /lib/systemd/system/

## Creating conda env
# Adding the environment files in the docker image
ADD base.yml base.yml
#ADD environment.yml environment.yml
#ADD environment_gcp.yml environment_gcp.yml
#ADD environment_2_0.yml environment_2_0.yml
#ADD environment_2_0_nightly.yml environment_2_0_nightly.yml

# Update conda manager
RUN conda update -n base conda -y

# Define a folder to store env
RUN mkdir -p /root/.conda-env/
RUN mkdir -p /home/.conda-env/
RUN mkdir -p /home/.conda-pkg/

# Create config files
RUN conda config --add channels conda-forge
RUN conda config --add envs_dirs /root/.conda-env/

# Update conda envs
RUN conda env update --file base.yml

# Create env
#RUN conda env create -f environment.yml
#RUN conda env create -f environment_gcp.yml
#RUN conda env create -f environment_2_0_nightly.yml

# Checking configuration and new created envs
RUN conda config --show
RUN conda config --show-sources
RUN conda config --show envs_dirs

RUN conda info
RUN conda info -e
RUN conda env list
RUN conda list

RUN ls -la /root/.conda-env/

# checking display conda env in notebook
RUN conda list nb_conda_kernels
RUN conda list notebook
RUN conda list ipykernel

#RUN conda list nb_conda_kernels -n env_gcp_dl
#RUN conda list notebook -n env_gcp_dl
#RUN conda list ipykernel -n env_gcp_dl

#RUN jupyter kernelspec list
#RUN jupyter --paths
#
### Clean all downloaded packages
#RUN conda clean -a -y
#
#### Check for update
#RUN jupyter labextension update --all
#### Check which extension exist
#RUN jupyter labextension list
##
## Update to fix issue with extension of jupyter lab
RUN jupyter labextension install @jupyterlab/github
RUN jupyter labextension install @jupyterlab/latex
RUN jupyter labextension install @mflevine/jupyterlab_html
#RUN jupyter labextension install jupyterlab-drawio !! issue !!
#RUN jupyter labextension install @jupyterlab/plotly-extension !! issue !!
#RUN jupyter labextension install jupyterlab_bokeh !! issue !!
RUN jupyter labextension install @jupyterlab/toc
RUN jupyter labextension install @aquirdturtle/collapsible_headings
RUN jupyter labextension install jupyterlab-jupytext
RUN jupyter labextension install jupyterlab-cpustatus
RUN jupyter labextension install jupyterlab-python-file
#RUN jupyter labextension install jupyterlab_toastify jupyterlab_conda !! issue !!
RUN jupyter labextension install @ijmbarr/jupyterlab_spellchecker
RUN jupyter labextension install @lckr/jupyterlab_variableinspector
RUN jupyter labextension install nbdime-jupyterlab
RUN jupyter labextension update @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension update @jupyterlab/celltags
RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter
#RUN jupyter labextension install @jupyterlab/jupyterlab-monaco  !! issue !!
#RUN jupyter labextension install jupyterlab-flake8 !! issue !!
RUN jupyter serverextension enable --py jupyterlab_code_formatter

RUN jupyter lab build

RUN jupyter labextension list

# create user and create home directory
#RUN useradd -m jupyter
#RUN pwd && cd /home &&  mkdir jupyter && ls -la && pwd && jupyter:users jupyter
#RUN adduser --disabled-password --gecos "" jupyter
#RUN useradd -rm -d /home/jupyter -s /bin/bash -g root -G sudo -u 1000 jupyter

# Checking more info
RUN id
RUN whoami
RUN cat /etc/passwd
RUN cat /etc/group

# Cheking is /home/jupyter was created
RUN ls -la
RUN ls -la  /home

# Trying to switch to jupyter user
#USER jupyter
#WORKDIR /home/jupyter

# Running some checks
RUN echo $HOME

# set the right permission
RUN chmod +x /entrypoint.sh

# Activate conda
RUN conda init bash && . /root/.bashrc && conda info -e # && conda activate base

## !!!!
## !!!! NO pip, conda, git command below because of proxy config !!!!
## !!!!

## Configuration related to the entrerpise proxy server
# Setup the proxy server for conda
ADD .condarc /.condarc
ENV CONDARC /.condarc

# drop local conda-env canal
RUN conda config --remove envs_dirs /root/.conda-env/

# Setup the proxy server for pip
ADD pip.conf /pip.conf
ENV PIP_CONFIG_FILE /pip.conf

# Setup the proxy server for git
ADD .gitconfig /root/.gitconfig
# Check that the config file was created properly
RUN git config --list
# Adding the config file to read entrerise Github
#RUN mkdir -p /root/.ssh/
#ADD known_hosts /root/.ssh/known_hosts
#ADD id_rsa /root/.ssh/id_rsa
#ADD config /root/.ssh/config

# Copy env if they exist from /root to /home
RUN sed -i '1 a cp -r /root/.conda-env/env* /home/.conda-env/.' /entrypoint.sh
RUN cat  /entrypoint.sh
# Running some checks about kernels
RUN sed -i '1 a if [ -e /home/config.sh ]\nthen\n    source /home/config.sh\nfi\njupyter serverextension list\nconda list nb_conda_kernels\nconda env list' /run_jupyter.sh
RUN cat  /run_jupyter.sh
# Adding option in jupyter notebook: display conda env, display hidden files
RUN sed -i -e "\$ac.NotebookApp.kernel_spec_manager_class = 'nb_conda_kernels.manager.CondaKernelSpecManager'" /root/.jupyter/jupyter_notebook_config.py
RUN sed -i -e "\$ac.ContentsManager.allow_hidden = True" /root/.jupyter/jupyter_notebook_config.py
RUN cat /root/.jupyter/jupyter_notebook_config.py
# Add execution of custom scripts it is exists
RUN sed -i -e "\$aif [ -e /home/config.sh ]\nthen\n    source /home/config.sh\nfi\n" /root/.bashrc
RUN cat /root/.bashrc

# New test
#RUN groupadd jupyter -g 1002
#RUN useradd -u 1001 -g 1002 --shell /bin/bash --create-home jupyter
#RUN chown jupyter:jupyter /pip.conf
#RUN chown jupyter:jupyter /root/.gitconfig
#RUN chown jupyter:jupyter /.condarc
#RUN chown jupyter:jupyter /entrypoint.sh
#RUN chown jupyter:jupyter /run_jupyter.sh
#RUN chown  -R jupyter:jupyter  /
#USER jupyter
#WORKDIR /home/jupyter

# Define entry points
ENTRYPOINT ["/entrypoint.sh", "/run_jupyter.sh"]
# For local test only
#ENTRYPOINT ["/entrypoint.sh", "/bin/bash"]

# Managed to have /home working
WORKDIR /home/

