FROM nvcr.io/nvidia/pytorch:22.07-py3
# FROM python:3.10.5-bullseye

COPY requirements.txt /tmp/pip-tmp/

RUN pip3 --no-cache-dir install 'git+https://github.com/facebookresearch/fvcore'
RUN pip3 --no-cache-dir install -r /tmp/pip-tmp/requirements.txt && rm -rf /tmp/pip-tmp

WORKDIR ./inference

COPY . .