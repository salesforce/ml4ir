# Using python based centos7 docker base image:
FROM ducouloa/ml4ir:0.1

# Set working directory and PYTHONPATH
USER root
WORKDIR /home/ml4ir/python/
RUN mkdir -p /home/ml4ir/python/
ENV PYTHONPATH=/home/ml4ir/python

# Install python dependencies
COPY requirements.txt /home/ml4ir/python/
RUN pip install --upgrade pip
RUN pip config set global.extra-index-url https://pypi.python.org/simple
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt

# Copy ml4ir
COPY . /home/ml4ir/python/

# Set docker entrypoint
RUN cp $PYTHONPATH/ml4ir/build/run_driver.sh $PYTHONPATH/.
RUN chmod +x run_driver.sh

ENTRYPOINT ["./run_driver.sh"]
