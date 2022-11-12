FROM ubuntu:18.04

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils

RUN apt-get update && apt-get install -y wget git pkg-config libprotobuf-dev protobuf-compiler libjson-c-dev intltool libx11-dev libxext-dev libjpeg-dev zlib1g-dev ninja-build

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7
RUN python --version

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

RUN git clone https://github.com/deepmind/spiral.git
WORKDIR /spiral
RUN git submodule update --init --recursive

RUN apt-get remove --purge cmake -y
RUN apt-get install software-properties-common -y
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' -y
RUN apt-get install cmake pkg-config protobuf-compiler libjson-c-dev intltool libpython3-dev python3-pip -y

RUN pip3 install six setuptools numpy scipy tensorflow==1.14 tensorflow-hub dm-sonnet==1.35
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

RUN python setup.py develop --user

RUN wget -c https://github.com/mypaint/mypaint-brushes/archive/v1.3.0.tar.gz -O - | tar -xz -C third_party \
    && git clone https://github.com/dli/paint third_party/paint \
    && patch third_party/paint/shaders/setbristles.frag third_party/paint-setbristles.patch

RUN pip3 install matplotlib jupyter

RUN apt-get autoremove -y && apt-get remove -y wget git pkg-config && apt-get autoremove -y

# FROM tensorflow/tensorflow:1.14.0-py3-jupyter
# FROM python:3.7

# RUN python --version

# RUN apt-get update && apt-get install -y wget git pkg-config libprotobuf-dev protobuf-compiler libjson-c-dev intltool libx11-dev libxext-dev libjpeg-dev zlib1g-dev ninja-build

# RUN pip3 install --no-cache-dir six scipy tensorflow-hub tensorflow-probability==0.7 dm-sonnet==1.35
# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.4/cmake-3.15.4-Linux-x86_64.sh \
#     -q -O /tmp/cmake-install.sh \
#     && chmod u+x /tmp/cmake-install.sh \
#     && mkdir /usr/bin/cmake \
#     && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
#     && rm /tmp/cmake-install.sh

# ENV PATH="/usr/bin/cmake/bin:${PATH}"

# RUN git clone https://github.com/deepmind/spiral.git
# WORKDIR /tf/spiral

# RUN pwd
# RUN git submodule update --init --recursive
# RUN wget -c https://github.com/mypaint/mypaint-brushes/archive/v1.3.0.tar.gz -O - | tar -xz -C third_party 
# RUN git clone https://github.com/dli/paint third_party/paint
# RUN patch third_party/paint/shaders/setbristles.frag third_party/paint-setbristles.patch

# COPY setup.patch setup.patch
# RUN patch setup.py setup.patch

# COPY cmakelists.patch cmakelists.patch
# RUN patch CMakeLists.txt cmakelists.patch

# RUN python3 setup.py develop --user

# RUN apt-get autoremove -y && apt-get remove -y wget git pkg-config && apt-get autoremove -y
