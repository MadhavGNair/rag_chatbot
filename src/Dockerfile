FROM public.ecr.aws/lambda/python:3.13

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install poetry (and ensure pip is up-to-date)
RUN pip install --upgrade pip poetry

# Install build dependencies and upgrade gcc (if needed)
RUN dnf update -y && \
    dnf install -y gcc gcc-c++ make cmake zip && \
    dnf clean all

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-root --without dev

# Copy application code
COPY . ./

# Set the command to run the application
CMD ["python", "/app/main.py"]

# Optional: For local testing
EXPOSE 8000