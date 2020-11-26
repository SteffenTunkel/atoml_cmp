FROM gradle:jdk11

# add files/folder to build the system
ADD src /usr/local/src
ADD testdata /usr/local/testdata
ADD build.gradle /usr/local/build.gradle
ADD settings.gradle /usr/local/settings.gradle

# change working directory
WORKDIR /usr/local

# build atoml
RUN gradle build

# add the bash script to execute the test generation (later add to have the build cached)
ADD docker_run_atoml_testgeneration.sh /usr/local

# run the bash script
CMD ["./docker_run_atoml_testgeneration.sh"]
