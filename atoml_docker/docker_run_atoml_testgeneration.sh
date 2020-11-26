#!/bin/sh

cd /usr/local/build/libs
echo create tests for sklearn:
java -jar atoml-0.1.0.jar -f /testdata/sklearn_descriptions.yml -nomorph -n 1000
echo create tests for spark:
java -jar atoml-0.1.0.jar -f /testdata/spark_descriptions.yml -nomorph -n 1000
echo create tests for weka:
java -jar atoml-0.1.0.jar -f /testdata/weka_descriptions.yml -nomorph -n 1000