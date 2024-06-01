FROM python:3.11.9

RUN mkdir -p /scratchml

WORKDIR /scratchml

RUN pip install --no-cache-dir -U pip

COPY . .

RUN pip install -r requirements/requirements_test.txt

CMD ["python3", "-m", "unittest", "discover"]