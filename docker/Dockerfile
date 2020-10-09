FROM python:3.8.6
ENV PYTHONUNBUFFERED=1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
COPY gunicorn.service /code/etc/systemd/system/gunicorn.service
COPY gunicorn.socket /code/etc/systemd/system/gunicorn.socket
