From python:3-onbuild

WORKDIR /test
ADD . /test

#RUN	pip install --upgrade pip && \
#	pip install -r requirements.txt

RUN pip install -r requirements.txt

CMD ["python", "./run_classification.py"]