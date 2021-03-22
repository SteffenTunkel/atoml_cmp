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
These instructions assume the use of the folder structure given by this repository.

1. Define the algorithms in a YAML file in the `algorithm-descriptions` folder fitting to the 
   [atoml test definition scheme](https://github.com/sherbold/atoml#definition-of-tests).
   Definitions with the same `name` and different `framework` will be compared with each other.
   Mind that atoml_cmp is not supporting the use of parameterized test cases. Therefore, this atoml feature
   should not been used since just the last parameter constellation is further evaluated.
   
2. Define datasets that are used for the test. This can be done in the directory of the atoml docker:
   `docker_collection/atoml_docker/`. In the `TestCatalogSelection.java` you have to add instances of the desired test
   classes to the `Smoketests` lists. For the comparative testing you can use smoke tests implemented in atoml,
   and use own `.arff` files to generate test cases. For that you can create a `SmoketestFromArff` instance. 
   The according files should be in `resources/` to be bound in the docker build.
2. Run the tool with by the command line: `python -m atoml_cmp`.
   
         optional arguments:
         -h, --help           show this help message and exit
         -d , --dockerlist    json file with docker descriptions ["dockerlist.json"]
         -y , --yaml_desc     directory of the yaml files with the algorithm definitions ["algorithm-descriptions"]
         -t , --testcases     directory for the generated test cases ["generated-tests"]
         -p , --predictions   directory for the prediction csv files ["predictions"]
         -a , --archive       directory, where to save the archive ["archive"]
   Alternatively the function `atoml_cmp.run_tool.main()` can be called.  

   The default procedure looks as follows: All docker defined by `dockerlist.json` are build and executed. 
   A `generated-tests` and a `predictions` directories are created to save test cases and test results (predictions),
   in case the directories exist before they get cleared. 
   The results of the comparison are then saved in the `archive` directory.

## Documentation

https://steffentunkel.github.io/atoml_cmp/

## License
atoml_cmp is licensed under the Apache License, Version 2.0.