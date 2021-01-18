FROM gradle:jdk11

# set working directory
WORKDIR /container

# copy files/folder to build the system
COPY src src
COPY testdata testdata
COPY build.gradle .
COPY settings.gradle .

# build atoml
RUN gradle build

# COPY the bash script to execute the test generation (later copy to have the build cached)
COPY docker_run_atoml_testgeneration.sh .

# run the bash script
CMD ["./docker_run_atoml_testgeneration.sh"]
