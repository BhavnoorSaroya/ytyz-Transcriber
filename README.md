# ytyz-Transcriber

### A powerful transcription application built in two parts
- a voice transcription tool meant to be used command line on via desktop
- a simple web wrapper written with fastapi that allows the transcription tool to be hosted in the cloud


### Setup and run instructions (need to improve)
- use uv (faster than pip)
- Tested running on ubuntu 24.04 with an rtx 3080
- Deps can get messy for older whisper models, requirements.txt is tested and known working
- Currently experimenting with fly.io gpu machines
- A token from hugging face is required (free but you must sign up)


### Improvements coming soon
- files stored in s3/tigris bucket, they are currently stored on the system
- ability to retrieve uploaded files and transcriptions by id (currently the system stores everything but only has an endpoint for the latest transcription)
- proper queue with ws implementation to notify completion (currently served by status endpoint that must be polled)
- dynamic gateway signatures to prevent unauthorized access (currently this is done by placing the service behind an auth + reverse proxy microservice)
