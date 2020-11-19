FROM python:3-onbuild

WORKDIR /test
ADD . /test

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "run_atoml_sklearn_testsuite.py"]