# debian Buster gives us  Python 3.6+
FROM debian:buster

HEALTHCHECK --interval=60s --timeout=5s --retries=3 --start-period=10s CMD wget --no-proxy -O - -q localhost:8080/ping

ENV http_proxy=http://128.7.3.56:3128
ENV https_proxy=http://128.7.3.56:3128

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
COPY GTETE_backend/models/de_core_news_sm-2.2.5.tar.gz  /usr/src/app/de_core_news_sm-2.2.5.tar.gz

# TODO find out  which of these is not necessarily required
ENV DEBIAN_FRONTEND noninteractive
#  it is important that the update and installation is in the same run statement
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    openssh-client \
    wget \
    # install python 3
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-virtualenv \
    python3-wheel \
    pkg-config \
    # install hunspell
    hunspell \
    libhunspell-dev \
    hunspell-de-de\
    # requirements for keras
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip==20.2
RUN pip3 --disable-pip-version-check install --proxy http://www-proxy.fkie.fraunhofer.de:3128 --no-cache-dir -r requirements.txt --ignore-installed PyYAML
RUN python3 -m spacy download de_core_news_sm --disable-pip-version-check
RUN pip3 --disable-pip-version-check install git+https://github.com/dtuggener/CharSplit.git


COPY . /usr/src/app

EXPOSE 8080

ENTRYPOINT ["python3"]

CMD ["-m", "GTETE_backend"]
