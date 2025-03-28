FROM python:3.13-slim

RUN apt-get update

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV AWS_LAMBDA_FUNCTION_HANDLER=lambda_function.lambda_handler

ENTRYPOINT ["python3", "-m", "awslambdaric"]

CMD ["lambda_function.lambda_handler"]