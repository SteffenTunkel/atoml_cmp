#!/bin/sh

#cd /usr/local/build/libs

for file in testdata/*.yml
do
echo create tests from "$file":
java -jar build/libs/atoml-0.1.0.jar -f "$file" -nomorph -n 10
done

# old version
#echo create tests for sklearn:
#java -jar atoml-0.1.0.jar -f /testdata/sklearn_descriptions.yml -nomorph -n 1000
#echo create tests for spark:
#java -jar atoml-0.1.0.jar -f /testdata/spark_descriptions.yml -nomorph -n 1000
#echo create tests for weka:
#java -jar atoml-0.1.0.jar -f /testdata/weka_descriptions.yml -nomorph -n 1000