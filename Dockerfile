FROM python:3.11.9

RUN mkdir -p /scratchml

WORKDIR /scratchml

RUN pip install --no-cache-dir -U pip

COPY . .

RUN pip install -r requirements/requirements.txt
