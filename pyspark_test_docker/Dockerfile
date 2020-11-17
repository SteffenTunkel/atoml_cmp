FROM jupyter/all-spark-notebook
MAINTAINER Manuel de la Peña <manuel.delapenya@liferay.com>

USER root

RUN cd /tmp && \
	rm -fr /usr/local/spark-* && \
	wget -q https://downloads.apache.org/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz && \
	tar xzf spark-3.0.1-bin-hadoop2.7.tgz -C /usr/local && \
	rm spark-3.0.1-bin-hadoop2.7.tgz
RUN cd /usr/local && rm spark && ln -s spark-3.0.1-bin-hadoop2.7 spark

RUN pip install runipy

ENV APACHE_SPARK_VERSION 3.0.1
ENV SPARK_HOME /usr/local/spark
ENV PYTHONPATH $SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip

COPY spark_wine_short.ipynb /home/jovyan/work/spark_wine_short.ipynb

#COPY wine_test_spark.csv /home/jovyan/work/wine_test_spark.csv
#COPY wine_train_spark.csv /home/jovyan/work/wine_train_spark.csv

RUN chown -R jovyan:users /home/jovyan/work/
CMD runipy /home/jovyan/work/spark_wine_short.ipynb