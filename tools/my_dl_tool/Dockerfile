FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
# Copy all files including the ONNX model
COPY . .

ENTRYPOINT ["python", "my_test_selector.py"]