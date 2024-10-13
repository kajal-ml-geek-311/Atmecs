from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import torch
import numpy as np
import torchxrayvision as xrv
from openai import OpenAI
from gtts import gTTS
import base64
import os
from dotenv import load_dotenv
import requests
import logging
from torchvision import transforms
from fastapi import Request
from typing import Optional

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
client = OpenAI(api_key=api_key)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Load the pre-trained DenseNet model with correct weights
try:
    model = xrv.models.DenseNet(weights="densenet121-res224-all")  # Use a valid weights string
    logging.info("DenseNet model loaded successfully with weights='densenet121-res224-all'.")
except Exception as e:
    logging.error(f"Error loading DenseNet model: {e}")
    raise

model = model.to(device)
model.eval()

# Verify the first layer to ensure it accepts 1 input channel
first_layer = model.features[0]
logging.info(f"First layer of the model: {first_layer}")
if first_layer.in_channels != 1:
    raise ValueError(f"Expected first layer to have in_channels=1, but got in_channels={first_layer.in_channels}.")

# Initialize the FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Allow CORS so frontend can communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': "Respond in English.",
    'hi': "Respond in Hindi.",
    # Add more languages and corresponding instructions as needed
}

# Updated Preprocessing function using torchvision transforms
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess the input PIL image:
    - Resize to 224x224
    - Convert to grayscale
    - Convert to tensor
    - Add batch dimension
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
        transforms.ToTensor(),  # Converts to [0, 1] and shapes to (C, H, W)
    ])
    
    image = transform(image)
    image_tensor = image.unsqueeze(0)  # Shape: (1, 1, 224, 224)
    logging.debug(f"Preprocessed image tensor shape: {image_tensor.shape}")
    return image_tensor

# Grad-CAM heatmap generation function (Placeholder)
def generate_gradcam_heatmap(image_tensor: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the input image_tensor using the specified model.
    This is a placeholder function. Implement Grad-CAM using libraries like torchcam for meaningful results.
    """
    # Placeholder: Return a dummy heatmap
    heatmap = np.random.rand(224, 224)
    return heatmap

# Diagnosis function with enhanced error handling and logging
def get_diagnosis(image: Image.Image, patient_data: str, language: str = 'en') -> tuple:
    """
    Perform diagnosis based on the input image and patient data.
    Returns:
        diagnosis (str): The diagnosis text.
        audio_bytes (bytes or None): The synthesized audio in the selected language.
        heatmap (np.ndarray or None): The Grad-CAM heatmap.
    """
    try:
        # Preprocess the image
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(device)
        logging.info(f"Image tensor moved to device: {image_tensor.device}")

        # Pass the image through the model
        with torch.no_grad():
            preds = model(image_tensor)
        logging.debug(f"Model predictions: {preds}")

        # Get predicted labels and probabilities
        labels = xrv.datasets.default_pathologies
        pred_probs = preds.cpu().numpy()[0]
        logging.debug(f"Prediction probabilities: {pred_probs}")

        # Get top 5 predictions
        top_indices = np.argsort(pred_probs)[::-1][:5]
        image_diagnosis = [(labels[i], pred_probs[i]) for i in top_indices]
        logging.info(f"Top predictions: {image_diagnosis}")

        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam_heatmap(image_tensor, model)
        logging.debug(f"Generated heatmap shape: {heatmap.shape}")

        # Prepare the prompt for GPT-4 with language specification
        lang_instruction = SUPPORTED_LANGUAGES.get(language.lower(), SUPPORTED_LANGUAGES['en'])
        image_analysis_text = "\n".join(
            [f"{condition}: {prob:.2f}" for condition, prob in image_diagnosis]
        )
        prompt = f"""
You are a medical AI assistant. {lang_instruction} Analyze the following medical image findings and patient data to provide a diagnosis.

Patient Data:
{patient_data}

Image Analysis Results:
{image_analysis_text}

Provide a detailed diagnosis and suggest possible treatment options.
"""

        logging.debug(f"LLM Prompt: {prompt}")

        # Call GPT-4 model using the OpenAI client
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        diagnosis = response.choices[0].message.content.strip()
        logging.info("LLM responded successfully.")

    except Exception as e:
        logging.error(f"Error in get_diagnosis: {e}", exc_info=True)
        diagnosis = f"An error occurred during diagnosis: {e}"
        return diagnosis, None, None

    # Generate audio from the diagnosis
    try:
        tts_language = 'en'  # Default to English
        if language.lower() == 'hi':
            tts_language = 'hi'

        tts = gTTS(text=diagnosis, lang=tts_language)
        tts_io = io.BytesIO()
        tts.write_to_fp(tts_io)
        tts_io.seek(0)

        # Read audio bytes
        audio_bytes = tts_io.read()
        logging.info("Audio generated successfully.")
    except Exception as e:
        logging.error(f"Error generating audio: {e}", exc_info=True)
        diagnosis += f"\n\nError generating audio: {e}"
        audio_bytes = None

    return diagnosis, audio_bytes, heatmap

# Synthetic Data Generation function with logging
def generate_synthetic_report(report_type: str, patient_age: int, patient_gender: str, medical_condition: str, language: str = 'en') -> str:
    """
    Generate a synthetic medical report based on the provided parameters.
    Returns:
        synthetic_report (str): The generated synthetic report.
    """
    try:
        lang_instruction = "Respond in Hindi." if language.lower() == 'hi' else "Respond in English."
        prompt = f"""
You are a medical AI assistant. {lang_instruction} Generate a synthetic {report_type.lower()} for a {patient_age}-year-old {patient_gender} patient diagnosed with {medical_condition}. Ensure that the report is realistic and useful for research and training purposes. Do not include any real patient data. Anonymize any identifiable information.
"""
        logging.debug(f"LLM Prompt for synthetic report: {prompt}")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant generating synthetic medical data."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.7,
        )
        synthetic_report = response.choices[0].message.content.strip()
        logging.info("Synthetic report generated successfully.")
    except Exception as e:
        logging.error(f"Error in generate_synthetic_report: {e}", exc_info=True)
        synthetic_report = f"An error occurred: {e}"
    
    return synthetic_report

# Treatment Plan Generation function with logging
def generate_treatment_plan(patient_data: str, language: str = 'en') -> str:
    """
    Generate a personalized treatment plan based on patient data.
    Returns:
        treatment_plan (str): The generated treatment plan.
    """
    try:
        lang_instruction = "Respond in Hindi." if language.lower() == 'hi' else "Respond in English."
        prompt = f"""
You are a medical AI assistant. {lang_instruction} Based on the following patient data, generate a personalized treatment plan. Reference relevant medical guidelines and literature where appropriate.

Patient Data:
{patient_data}

Provide a detailed treatment plan, including medication recommendations, lifestyle changes, and any necessary follow-up tests or consultations.
"""
        logging.debug(f"LLM Prompt for treatment plan: {prompt}")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant generating personalized treatment plans based on patient data and medical literature."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.7,
        )
        treatment_plan = response.choices[0].message.content.strip()
        logging.info("Treatment plan generated successfully.")
    except Exception as e:
        logging.error(f"Error in generate_treatment_plan: {e}", exc_info=True)
        treatment_plan = f"An error occurred: {e}"
    
    return treatment_plan

# Endpoint for Diagnosis with enhanced error handling
@app.post("/diagnosis")
async def diagnosis_endpoint(
    image: UploadFile = File(...),
    patient_data: str = Form(...),
    language: str = Form('en')  # Default to English
):
    """
    Endpoint to perform diagnosis based on an uploaded image and patient data.
    Accepts:
        - image: UploadFile (JPEG or PNG)
        - patient_data: str
        - language: str ('en' or 'hi')
    Returns:
        - diagnosis: str
        - audio: base64 encoded audio string
    """
    # Validate language
    if language.lower() not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}'. Supported languages are: {', '.join(SUPPORTED_LANGUAGES.keys())}."
        )
    
    # Validate image type
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only JPEG and PNG are supported."
        )
    
    try:
        # Read and open the image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert('L')  # Ensure grayscale
        logging.info("Image uploaded and converted to grayscale successfully.")
    except Exception as e:
        logging.error(f"Error processing uploaded image: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {e}"
        )
    
    # Get diagnosis, audio, and heatmap
    diagnosis_text, audio_bytes, heatmap = get_diagnosis(image_pil, patient_data, language)
    
    # Encode audio bytes to base64
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else None
    
    return {"diagnosis": diagnosis_text, "audio": audio_b64}

# Endpoint for Synthetic Report Generation with enhanced error handling
@app.post("/generate_report")
async def generate_report_endpoint(
    report_type: str = Form(...),
    patient_age: int = Form(...),
    patient_gender: str = Form(...),
    medical_condition: str = Form(...),
    language: str = Form('en')  # Default to English
):
    """
    Endpoint to generate a synthetic medical report.
    Accepts:
        - report_type: str
        - patient_age: int
        - patient_gender: str
        - medical_condition: str
        - language: str ('en' or 'hi')
    Returns:
        - report: str
    """
    # Validate language
    if language.lower() not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}'. Supported languages are: {', '.join(SUPPORTED_LANGUAGES.keys())}."
        )
    
    synthetic_report = generate_synthetic_report(report_type, patient_age, patient_gender, medical_condition, language)
    return {"report": synthetic_report}

# Endpoint for Treatment Plan Generation with enhanced error handling


@app.post("/treatment_plan")
async def treatment_plan_endpoint(
    request: Request,
    patient_name: str = Form(...),
    patient_age: int = Form(...),
    patient_gender: str = Form(...),
    medical_condition: str = Form(...),
    current_medications: Optional[str] = Form(None),
    allergies: Optional[str] = Form(None),
    language: str = Form('en')  # Default to English
):
    try:
        logging.info(f"Received treatment plan request from {request.client.host}")
        logging.debug(f"Patient Name: {patient_name}")
        logging.debug(f"Patient Age: {patient_age}")
        logging.debug(f"Patient Gender: {patient_gender}")
        logging.debug(f"Medical Condition: {medical_condition}")
        logging.debug(f"Current Medications: {current_medications}")
        logging.debug(f"Allergies: {allergies}")
        logging.debug(f"Language: {language}")

        # Validate language
        if language.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{language}'. Supported languages are: {', '.join(SUPPORTED_LANGUAGES.keys())}."
            )

        # Construct patient_data
        patient_data = f"""
        Name: {patient_name}
        Age: {patient_age}
        Gender: {patient_gender}
        Medical Condition: {medical_condition}
        Current Medications: {current_medications or 'None'}
        Allergies: {allergies or 'None'}
        """

        # Generate treatment plan
        treatment_plan = generate_treatment_plan(patient_data, language)
        logging.info("Treatment plan generated successfully.")
        return {"treatment_plan": treatment_plan}

    except Exception as e:
        logging.error(f"Error in treatment_plan_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
