FROM openjdk:8-jdk

WORKDIR /container

# Install maven
RUN apt-get update
RUN apt-get install -y maven

# Prepare by downloading dependencies
ADD pom.xml .
RUN ["mvn", "dependency:resolve"]
RUN ["mvn", "verify"]


CMD ["mvn", "test"]