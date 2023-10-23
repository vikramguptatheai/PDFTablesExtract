FROM python:3.10
RUN apt-get update && apt-get -y install ghostscript && apt-get clean
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 8080
WORKDIR /PDFTablesExtract
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "PDFTables.py", "--server.port=8081", "--server.address=0.0.0.0"]
