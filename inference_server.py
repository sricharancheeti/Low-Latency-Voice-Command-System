import torch
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import asyncio
from typing import List

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased" # Or your fine-tuned model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH_SIZE = 16
BATCH_TIMEOUT = 0.1 # seconds

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Global Variables ---
# Load the fine-tuned and quantized model and tokenizer
# In a real-world scenario, the model would be loaded from a saved fine-tuned version.
# For this PoC, we load the base model with quantization.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2, # Example: 2 labels (e.g., 'command', 'not_command')
    load_in_8bit=True # Apply 8-bit quantization on load
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval() # Set model to evaluation mode

# Queue to hold incoming requests for batching
request_queue = asyncio.Queue()

# --- Background Worker for Batch Processing ---
async def batch_processing_worker():
    """
    This worker runs in the background, continuously collecting requests
    from the queue and processing them in batches.
    """
    while True:
        requests_to_process = []
        
        # Wait for the first request
        first_req = await request_queue.get()
        requests_to_process.append(first_req)

        # Collect more requests for a short period (BATCH_TIMEOUT)
        # to form a larger batch, up to MAX_BATCH_SIZE
        start_time = asyncio.get_event_loop().time()
        while (
            len(requests_to_process) < MAX_BATCH_SIZE and
            (asyncio.get_event_loop().time() - start_time) < BATCH_TIMEOUT
        ):
            try:
                req = request_queue.get_nowait()
                requests_to_process.append(req)
            except asyncio.QueueEmpty:
                break # No more requests in the queue, process what we have

        # Prepare batch for the model
        texts = [req["text"] for req in requests_to_process]
        
        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Set results for each future object
            for i, req in enumerate(requests_to_process):
                req["future"].set_result(predictions[i].tolist())

        except Exception as e:
            # If an error occurs, set the exception for all futures in the batch
            for req in requests_to_process:
                req["future"].set_exception(e)


@app.on_event("startup")
async def startup_event():
    """
    On startup, create the background worker task.
    """
    asyncio.create_task(batch_processing_worker())


@app.post("/predict")
async def predict(request: Request):
    """
    API endpoint to receive a single prediction request.
    It adds the request to the queue and waits for the result.
    """
    try:
        data = await request.json()
        text_input = data["text"]
        
        future = asyncio.Future()
        
        await request_queue.put({
            "text": text_input,
            "future": future
        })
        
        result = await future
        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}, 500
