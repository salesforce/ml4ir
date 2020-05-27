# Using python based centos7 docker base image:
ARG BASE_IMAGE
FROM $BASE_IMAGE:latest

USER root
WORKDIR /usr/lib

# yum installations 
RUN yum -y update
RUN yum -y install git sfdc-python37-devel.x86_64 gcc hadoop which krb5-workstation gcc-c++ java-1.8.0-openjdk-devel

# Set up pip
RUN pip install --upgrade pip

# Environment variables setup
ENV JAVA_HOME=/usr/lib/jvm/java
ENV HADOOP_VERSION=2.7.2
ENV HADOOP_HOME=/usr/local
ENV HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin

# System configurations
# Dockerfile currently has no easy way to do conditional copy.
# So we copy .env as a dummy file
COPY .env common/resources/mapred-site.xml? $HADOOP_CONF_DIR/.
COPY .env common/resources/core-site.xml? $HADOOP_CONF_DIR/.
COPY .env common/resources/hdfs-site.xml? $HADOOP_CONF_DIR/.
COPY .env common/resources/krb5.conf? /etc/.

# You can either clone a new branch from the public repository or use local branch
RUN mkdir -p /home/jenkins/ml4ir/
RUN if [[ "$CLONE_GIT_REPO" -eq "1" ]]; then git clone -b ${GIT_BRANCH} https://github.com/salesforce/ml4ir.git /home/jenkins/ml4ir; fi
# The below command is used to copy the contents of a local branch if CLONE_GIT_REPO != 1
# NOTE: This requires that ml4ir is the current directory
COPY . /home/jenkins/ml4ir/

# Set PYTHONPATH environment variable
ENV PYTHONPATH=/home/jenkins/ml4ir/python
WORKDIR $PYTHONPATH

# Python dependencies
RUN pip config set global.extra-index-url https://pypi.python.org/simple
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt

# Entrypoint
RUN cp $PYTHONPATH/ml4ir/build/run_driver.sh $PYTHONPATH/.
RUN chmod +x run_driver.sh

ENTRYPOINT ["./run_driver.sh"]
