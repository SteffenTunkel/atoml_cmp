FROM gradle:jdk11

COPY . /usr/local

WORKDIR /usr/local


#CMD ["gradlew","run",'--args="-f testdata/crossref_descriptions.yml -nomorph"']

#CMD gradlew run --args="-f testdata/crossref_descriptions.yml -nomorph"

RUN gradle build

CMD gradle run --args="-f /testdata/crossref_descriptions.yml --instances 1000 -nomorph "

