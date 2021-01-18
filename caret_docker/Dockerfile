#FROM r-base:4.0.3
FROM rocker/r-ver:4.0.0

WORKDIR  /home

## copy files
COPY install_packages.R /home/install_packages.R
COPY test_runner.R /home/test_runner.R


## install R-packages
RUN Rscript install_packages.R

#COPY /tests /home/tests

## run test folder
CMD Rscript test_runner.R