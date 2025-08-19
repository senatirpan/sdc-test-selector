FROM python:3.11

WORKDIR /app

# Install required packages
RUN pip install grpcio-tools
RUN pip install onnxruntime
RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn

# Copy all files including the ONNX model
COPY . .

ENTRYPOINT ["python", "my_test_selector.py"]