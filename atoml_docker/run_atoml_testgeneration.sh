#!/bin/sh

for file in testdata/*.yml
do
echo create tests from "$file":
java -jar build/libs/atoml-0.1.0.jar -f "$file" -nomorph -n 100 -predictions -t 1000
done