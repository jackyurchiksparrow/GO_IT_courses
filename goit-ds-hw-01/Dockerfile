# first, set the base image and build everything else on top of it
# instead of pulling the entire python image (1gb+), we target the main functionality in the smaller distributions. "alpine" (~50mb) doesn't cover pandas, numpy etc. because it uses a different C-library, the best choice is "slim" (~120-150mb)
# set the same version as in Poetry to avoid "Heisenbugs" (only arising in Docker)
FROM python:3.11-slim

# Create a folder called `/app` and switch to it (not to clutter linux system which has everything organized in a few folders)
WORKDIR /app

# ----------
# IMPORTANT: Docker runs every line in order so we first install and copy things that won't be changed (saves time and resources when editing files and having to reassemble the image)
# ----------

# if you don't use Poetry, the line can be:
# RUN pip install -r requirements.txt
RUN pip install poetry==2.3.4

# Copy dependency files ONLY (again, saves time and resources when reassembling the image)
COPY pyproject.toml poetry.lock* ./

# tell Poetry to install the libraries with no virtual environment (docker takes care of that)
# --no-root says "just install the libraries, don't try to install my `goit-ds-hw-01` project as a package"
RUN poetry config virtualenvs.create false && poetry install --no-root

# Copy THE REST of the app into workdir
# The first `.` means "The folder on my computer where the Dockerfile is."
# The second `.` means "The current folder inside the Docker container" (which is `/app` if you set the `WORKDIR`).
COPY . .

# run the app
CMD ["python", "main.py"]