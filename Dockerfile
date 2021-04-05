FROM python:3.7

RUN pip install pipenv==2020.11.15
ENV PIPENV_VENV_IN_PROJECT="True"

WORKDIR /opt/program
COPY Pipfile Pipfile.lock /opt/program/
RUN pipenv install --system

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY main.py /opt/program/
COPY lib /opt/program/lib/