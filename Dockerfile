FROM python:3.9

WORKDIR /opt/app
COPY requirements.txt /opt/app/requirements.txt
RUN pip3 install -r requirements.txt

COPY bot.py /opt/app

ENTRYPOINT "python3"
CMD "/opt/app/bot.py"
