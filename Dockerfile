# use official python image as parent image
FROM python:3.12-slim

#set the working directory in container
WORKDIR /app

#Copy the folder in to the container working directory
COPY . /app

# install requirements.txt 
RUN pip install -r requirements.txt

#Run the main.py to run the project
CMD ["python","main.py"]