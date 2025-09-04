Use an official Python runtime as a parent image
FROM python:3.9-slim

Set the working directory in the container
WORKDIR /app

Copy the dependencies file to the working directory
COPY requirements.txt .

Install any needed packages specified in requirements.txt
We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

Copy the content of the local src directory to the working directory
COPY . .

Specify the command to run on container startup
This will run the Uvicorn server, making the API available
CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "80"]