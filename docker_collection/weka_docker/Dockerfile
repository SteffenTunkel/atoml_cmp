FROM openjdk:8-jdk

WORKDIR /container

# Install maven
RUN apt-get update
RUN apt-get install -y maven

# Pre-build with maven
COPY pom.xml .
RUN ["mvn", "dependency:resolve"]
RUN ["mvn", "verify"]

# Run tests
COPY run_tests.sh .
ENTRYPOINT ["bash", "run_tests.sh"]
