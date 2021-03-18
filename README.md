# atoml_cmp
![CI](https://github.com/steffentunkel/atoml_cmp/workflows/CI/badge.svg)

Comparative testing of machine learning frameworks based on [atoml](https://github.com/sherbold/atoml).
## Description
A tool for the comparative testing of different machine learning frameworks.
The tool offers functionalities to set up and run the test case and test data generation based on atoml 
and also the test execution in a containerized manner.
The outcome of the test are given with the predictions for each algorithm implementation on each test case dataset.
Additionally, the tool offers an evaluation pipeline to actually compare the predictions of the different machine
learning algorithms. 

Through the use of docker container the requirements to generate and execute tests for various machine learning
frameworks are just a docker installation and python.  

Currently, the tool is supporting Scikit-Learn (Python), Weka (Java), Apache Spark MLLib (Scala) and Caret (R).

## Instructions
This instructions assume the use of the folder structure given by this repository.

1. Define the algorithms in a YAML file in the `algorithm-descriptions` folder fitting to the 
   [atoml test definition scheme](https://github.com/sherbold/atoml#definition-of-tests).
   Definitions with the same `name` and different `framework` will be compared with each other.
   Mind that atoml_cmp is not supporting the use of parameterized test cases. Therefore, this atoml feature
   should not been used since just the last parameter constellation is further evaluated.
   
2. Run the tool with:
   
        python -m atoml_cmp.run_tool
   
   All docker defined by `dockerlist.json` are build and executed. 
   By default, a `generated-tests` and a `predictions` directories are created to save test cases and test results (predictions).
   The comparison evaluation results are then saved in the `archive` directory.


## ToDo
- restructure the external data bind in
- fix linux issues with external data set use.
- better structure for function calls 

## Documentation

https://steffentunkel.github.io/atoml_cmp/

## License
atoml_cmp is licensed under the Apache License, Version 2.0.