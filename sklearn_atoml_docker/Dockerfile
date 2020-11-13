From python:3-onbuild

WORKDIR /test
ADD . /test

#RUN	pip install --upgrade pip && \
#	pip install -r requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "./tests/test_SKLEARN_GaussianNB.py"]