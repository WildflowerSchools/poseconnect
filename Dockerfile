FROM python:3.8
RUN apt-get update
RUN apt-get -y -qq install libgl1-mesa-glx
WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir .
