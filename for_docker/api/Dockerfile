FROM lyangasovid/myrepo:neurotext_without_run

# Copy the current directory contents into the container at /app
ADD . .
WORKDIR .
#FROM without_run_without_port:latest
# Make port 80 available to the world outside this container
EXPOSE 5080

ENTRYPOINT ["python", "-u", "neurotext.py"]
