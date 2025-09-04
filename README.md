Low-Latency Voice Command System (Proof-of-Concept)
This project is a proof-of-concept for a high-performance, low-latency voice command system. It's designed to serve a fine-tuned, quantized language model capable of handling millions of daily requests with a sub-500ms inference time, as described in the Wispr ML Engineer role.The core of the project is a custom Python inference server built with FastAPI that serves a distilled and 8-bit quantized version of a base language model.
Architectural Overview
Model Distillation & Fine-Tuning: We start with a pre-trained base model (e.g., a small variant like distilbert-base-uncased). This model is then fine-tuned on a specialized dataset of voice commands and then distilled into a smaller, faster version.
Quantization: To further accelerate inference speed and reduce the memory footprint, the fine-tuned model is loaded with 8-bit quantization using the bitsandbytes library.
Inference Server: A custom asynchronous API is built using FastAPI to handle incoming requests. It includes logic for batching multiple incoming requests together to maximize hardware utilization (GPU).Containerization: The entire application is containerized using Docker, ensuring a consistent, reproducible environment for both development and deployment.
How to Run This Project:
1.Clone the repository:git clone Low-Latency-Voice-Command-System
cd Low-Latency-Voice-Command-System
2. Install dependencies:pip install -r requirements.txt
(Ensure you have PyTorch with CUDA support if you are using a GPU)
3. (Optional) Run the model preparation script:This step is for fine-tuning and saving your own model. A pre-trained placeholder is used by default.python model_utils.py
4. Run the inference server:uvicorn inference_server:app --reload

