FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
COPY . .
RUN pip install -r requirements.txt
CMD ["/bin/bash"]
#