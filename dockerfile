FROM python:3.8.0-buster

WORKDIR /app

COPY . .

RUN pip install Cython

RUN pip install -r requirements.txt


# docker build -t python3-11-2 .

#Windows CMD
# docker run -idt --name python3-11-2 -v %cd%:/app python3-11-2
#Windows PowerShell、 Linux、Mac
# docker run -idt --name python3-11-2 -v ${PWD}:/app python3-11-2

# docker exec -it python3-11-2 /bin/bash

# pip freeze --all > requirements.txt