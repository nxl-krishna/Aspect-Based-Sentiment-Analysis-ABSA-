FROM python:3.9-slim

# Create a non-root user (Hugging Face Spaces requirement for Docker)
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Set up environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the project files into the container
COPY --chown=user . $HOME/app

# Install Python dependencies
# requirements.txt already includes the CPU-only PyTorch index URL
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Run the FastAPI server on port 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
