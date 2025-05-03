FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Install the package in editable mode from src directory
RUN cd src && pip install -e .

CMD ["python", "external_infer.py"]
