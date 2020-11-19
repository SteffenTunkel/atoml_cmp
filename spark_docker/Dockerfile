#FROM java:8 
FROM openjdk:8-jdk

# Install maven
RUN apt-get update
RUN apt-get install -y maven

WORKDIR /code

# Prepare by downloading dependencies
ADD pom.xml /code/pom.xml
RUN ["mvn", "dependency:resolve"]
RUN ["mvn", "verify"]

# Adding source, compile and package into a fat jar
ADD src /code/src

# compile
#RUN ["mvn", "compile"]

# Run the tests in the run of the dockerfile
#CMD ["mvn", "package"]
CMD ["mvn", "test"]


#RUN ["mvn", "package"]
#EXPOSE 4567
#CMD ["/usr/lib/jvm/java-8-openjdk-amd64/bin/java", "-jar", "target/sparkexample-jar-with-dependencies.jar"]