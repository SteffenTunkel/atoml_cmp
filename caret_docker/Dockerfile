#FROM r-base:4.0.3
FROM rocker/r-ver:4.0.0

WORKDIR  /container

## copy files
COPY install_packages.R /container/install_packages.R



## install R-packages
RUN Rscript install_packages.R

#COPY /tests /home/tests
COPY test_runner.R /container/test_runner.R

## run test folder
CMD Rscript test_runner.R