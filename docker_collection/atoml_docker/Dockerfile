FROM openjdk:11-jdk

# get repo of atoml
WORKDIR /
RUN apt-get update && apt-get install -y git
RUN git clone -b master https://github.com/SteffenTunkel/atoml.git container

# set actual working directory
WORKDIR /container

# change TestCatalog.java in atoml code (to choose the test data)
COPY TestCatalogSelection.java src/main/java/atoml/testgen/TestCatalog.java

# add external datasets to atoml resources
COPY resources/ src/main/resources/

# build atoml
RUN chmod u+x gradlew
RUN ./gradlew build

# COPY the bash script to execute the test generation
COPY run_atoml_testgeneration.sh .

# run the bash script
ENTRYPOINT ["bash", "run_atoml_testgeneration.sh"]
CMD ["100"]
