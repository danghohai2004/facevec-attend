FROM ubuntu:latest
LABEL authors="haico"

ENTRYPOINT ["top", "-b"]