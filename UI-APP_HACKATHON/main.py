 
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Body
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
import pytesseract
import pdf2image
import pydicom
import nibabel as nib
import pandas as pd
from typing import Optional, List, Dict
from pydantic import BaseModel
import json
import logging
from datetime import datetime
from torchvision import transforms

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported languages
SUPPORTED_LANGUAGES = {
    'en-GB': "Respond in English.",
    'hi': "Respond in Hindi.",
    'te': "Respond in Telugu."
}

# Set up patient data directory
PATIENT_DATA_DIR = 'patient_data'
os.makedirs(PATIENT_DATA_DIR, exist_ok=True)

class MedicalAI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.models = self._load_models()
        self.supported_formats = {
            'xray': ['.dcm', '.png', '.jpg', '.jpeg'],
            'mri': ['.nii', '.nii.gz', '.dcm'],
            'ct': ['.dcm', '.nii', '.nii.gz'],
            'reports': ['.pdf', '.png', '.jpg', '.jpeg'],
            'blood_work': ['.pdf', '.csv', '.xlsx']
        }

    def _load_models(self) -> Dict:
        """Load all required AI models"""
        try:
            models = {
                'xray': xrv.models.DenseNet(weights="densenet121-res224-all").to(self.device)
            }
            models['xray'].eval()
            return models
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    async def process_medical_file(self, file: UploadFile, file_type: str) -> dict:
        """Process different types of medical files"""
        content = await file.read()
        ext = os.path.splitext(file.filename)[1].lower()
        
        try:
            if file_type == 'xray' and ext in self.supported_formats['xray']:
                return await self._process_xray(content)
            elif file_type in ['mri', 'ct'] and ext in self.supported_formats[file_type]:
                return await self._process_scan(content, file_type)
            elif file_type == 'blood_work' and ext in self.supported_formats['blood_work']:
                return await self._process_blood_work(content, ext)
            elif file_type == 'reports' and ext in self.supported_formats['reports']:
                return await self._process_medical_report(content, ext)
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file format {ext} for {file_type}"
                )
        except Exception as e:
            logger.error(f"Error processing {file_type} file: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _process_xray(self, content: bytes) -> dict:
        """Process X-ray images"""
        try:
            image = Image.open(io.BytesIO(content)).convert('L')
            tensor = self._preprocess_image(image)
            
            with torch.no_grad():
                preds = self.models['xray'](tensor)
            
            labels = xrv.datasets.default_pathologies
            pred_probs = preds.cpu().numpy()[0]
            
            findings = [
                {
                    "condition": label,
                    "probability": float(prob),
                    "significance": "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"
                }
                for label, prob in zip(labels, pred_probs)
                if prob > 0.1  # Only include findings with >10% probability
            ]
            
            return {
                'type': 'xray',
                'findings': findings,
                'analysis_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing X-ray: {e}")
            raise

    async def _process_scan(self, content: bytes, scan_type: str) -> dict:
        """Process MRI or CT scan data"""
        try:
            # Save temporary file
            temp_file = f"temp_{scan_type}_{datetime.now().timestamp()}"
            with open(temp_file, 'wb') as f:
                f.write(content)
            
            if scan_type == 'mri':
                scan_data = nib.load(temp_file)
            else:  # CT scan
                scan_data = pydicom.dcmread(temp_file)
            
            # Process scan data and generate findings
            # This is a placeholder for actual scan processing logic
            findings = {
                'type': scan_type,
                'scan_date': datetime.now().isoformat(),
                'findings': f"Detailed {scan_type} analysis would go here",
                'recommendations': []
            }
            
            # Cleanup
            os.remove(temp_file)
            
            return findings
        except Exception as e:
            logger.error(f"Error processing {scan_type}: {e}")
            raise

    async def _process_blood_work(self, content: bytes, ext: str) -> dict:
        """Process blood work reports"""
        try:
            if ext == '.pdf':
                text = await self._extract_text_from_pdf(content)
            elif ext in ['.csv', '.xlsx']:
                df = pd.read_csv(io.BytesIO(content)) if ext == '.csv' else pd.read_excel(io.BytesIO(content))
                text = df.to_string()
            
            analysis = await self._analyze_blood_work(text)
            return {
                'type': 'blood_work',
                'findings': analysis,
                'analysis_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing blood work: {e}")
            raise

    async def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF files"""
        try:
            images = pdf2image.convert_from_bytes(content)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    async def _analyze_blood_work(self, text: str) -> dict:
        """Analyze blood work results using GPT-4"""
        try:
            prompt = f"""
            Analyze the following blood work results and provide:
            1. Abnormal values and their clinical significance
            2. Overall health assessment
            3. Recommended follow-up tests if any
            4. Lifestyle and dietary recommendations
            
            Blood Work Results:
            {text}
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical expert specialized in laboratory results interpretation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return {
                'analysis': response.choices[0].message.content,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing blood work: {e}")
            raise

    async def generate_diagnosis(self, patient_data: dict, medical_files: List[dict]) -> dict:
        """Generate comprehensive diagnosis"""
        try:
            # Compile all findings
            compiled_data = self._compile_medical_data(patient_data, medical_files)
            
            prompt = f"""
            Based on the following patient data and medical findings, provide a comprehensive diagnosis:
            
            Patient Information:
            {json.dumps(patient_data, indent=2)}
            
            Medical Findings:
            {json.dumps(compiled_data, indent=2)}
            
            Please provide:
            1. Primary diagnosis
            2. Differential diagnoses
            3. Supporting evidence
            4. Recommended additional tests
            5. Risk factors
            6. Prognosis
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert medical AI system providing comprehensive diagnoses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return {
                'diagnosis': response.choices[0].message.content,
                'timestamp': datetime.now().isoformat(),
                'data_sources': [file.get('type') for file in medical_files]
            }
        except Exception as e:
            logger.error(f"Error generating diagnosis: {e}")
            raise

    def _compile_medical_data(self, patient_data: dict, medical_files: List[dict]) -> dict:
        """Compile and organize all medical data"""
        compiled_data = {
            'xray_findings': [],
            'scan_findings': [],
            'blood_work_results': [],
            'other_findings': []
        }
        
        for file in medical_files:
            file_type = file.get('type')
            if file_type == 'xray':
                compiled_data['xray_findings'].append(file.get('findings'))
            elif file_type in ['mri', 'ct']:
                compiled_data['scan_findings'].append(file.get('findings'))
            elif file_type == 'blood_work':
                compiled_data['blood_work_results'].append(file.get('findings'))
            else:
                compiled_data['other_findings'].append(file)
        
        return compiled_data

# Initialize FastAPI app and medical AI system
app = FastAPI(title="Medical AI Assistant")
medical_ai = MedicalAI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_patient_file_path(patient_id: str) -> str:
    """Get the file path for patient data"""
    return os.path.join(PATIENT_DATA_DIR, f'patient_{patient_id}.json')

def load_patient_history(patient_id: str) -> List[dict]:
    """Load patient conversation history"""
    file_path = get_patient_file_path(patient_id)
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('conversation', [])
    except json.JSONDecodeError:
        return []

# Helper function for language validation
def get_language_name(code: str) -> str:
    """Convert language code to full name"""
    language_names = {
        'en-GB': 'English',
        'hi': 'Hindi',
        'te': 'Telugu'
    }
    return language_names.get(code, code)

def save_patient_history(patient_id: str, conversation: list):
    """Save patient conversation history"""
    file_path = get_patient_file_path(patient_id)
    with open(file_path, 'w') as f:
        json.dump({'conversation': conversation}, f, indent=4)

def append_to_patient_history(patient_id: str, role: str, content: str):
    """Append new message to patient history"""
    conversation = load_patient_history(patient_id)
    conversation.append({
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    })
    save_patient_history(patient_id, conversation)

class FileUploadRequest(BaseModel):
    patient_id: str
    file_type: str

class DiagnosisRequest(BaseModel):
    patient_id: str
    patient_data: dict
    medical_files: List[dict]

class ChatbotRequest(BaseModel):
    patient_id: str
    message: str
    language: str = 'en'

@app.post("/upload_medical_file")
async def upload_medical_file(
    file: UploadFile,
    file_type: str = Form(...),
    patient_id: str = Form(...)
):
    """Upload and process medical files"""
    try:
        result = await medical_ai.process_medical_file(file, file_type)
        append_to_patient_history(
            patient_id,
            'system',
            f"Processed {file_type} file: {file.filename}"
        )
        return result
    except Exception as e:
        logger.error(f"Error processing medical file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diagnosis")
async def get_diagnosis(request: DiagnosisRequest):
    """Get comprehensive diagnosis"""
    try:
        diagnosis = await medical_ai.generate_diagnosis(
            request.patient_data,
            request.medical_files
        )
        append_to_patient_history(
            request.patient_id,
            'system',
            f"Generated diagnosis based on {len(request.medical_files)} medical files"
        )
        return diagnosis
    except Exception as e:
        logger.error(f"Error getting diagnosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chatbot")
async def chatbot_endpoint(request: ChatbotRequest):
    """
    Chatbot endpoint supporting English, Hindi, and Telugu
    """
    try:
        logger.info(f"Received chatbot request - Patient ID: {request.patient_id}, Language: {request.language}")
        
        # Normalize language code to lowercase
        language = request.language
        
        # Validate language
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language requested: {language}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Unsupported language",
                    "message": "Supported languages are: English (en), Hindi (hi), and Telugu (te)",
                    "supported_languages": list(SUPPORTED_LANGUAGES.keys())
                }
            )

        # Load conversation history
        conversation = load_patient_history(request.patient_id)
        
        # Prepare system prompt with language instruction
        lang_instruction = SUPPORTED_LANGUAGES[language]
        system_prompt = f"""You are a medical AI assistant who is an expert in various medical fields. {lang_instruction}

Your capabilities include:
- Diagnosing medical conditions based on symptoms
- Explaining medical procedures and treatments
- Providing general health advice
- Answering questions about medications
- Offering preventive healthcare recommendations

Remember to:
1. Always maintain medical accuracy
2. Be clear and empathetic in your responses
3. Suggest consulting a healthcare provider when appropriate
4. Keep responses culturally appropriate for the language being used
5. Always prescribe remedy or Ayurvedic remedies
6. Always prescribe some meal plans and course of actions
7. also suggest which type of yoga and exercise could help
"""
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add recent conversation history (last 5 messages)
        for msg in conversation[-5:]:
            if msg['role'] != 'system':
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        # Get response from GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Save to history
        append_to_patient_history(request.patient_id, 'user', request.message)
        append_to_patient_history(request.patient_id, 'assistant', ai_response)
        
        # Generate audio response
        audio_bytes = None
        try:
            # Map language codes for gTTS
            tts_language_map = {
                'en-GB': 'en-GB',
                'hi': 'hi',
                'te': 'te'
            }
            
            tts_language = tts_language_map.get(language, 'en-GB')
            tts = gTTS(text=ai_response, lang=tts_language)
            audio_io = io.BytesIO()
            tts.write_to_fp(audio_io)
            audio_io.seek(0)
            audio_bytes = base64.b64encode(audio_io.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Error generating audio response: {e}")
            
        logger.info(f"Successfully generated response for Patient ID: {request.patient_id}")
        
        return {
            "response": ai_response,
            "audio": audio_bytes,
            "language": language
        }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in chatbot endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred while processing your request"
            }
        )

class TreatmentPlanRequest(BaseModel):
    patient_id: str
    diagnosis: dict
    patient_data: dict
    language: str = 'en-GB'

@app.post("/treatment_plan")
async def get_treatment_plan(request: TreatmentPlanRequest):
    """Generate personalized treatment plan"""
    try:
        # Validate language
        if request.language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Supported languages are: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )
        
        lang_instruction = SUPPORTED_LANGUAGES[request.language]
        
        prompt = f"""
        {lang_instruction}
        
        Based on the following diagnosis and patient data, generate a comprehensive treatment plan:
        
        Patient Information:
        {json.dumps(request.patient_data, indent=2)}
        
        Diagnosis:
        {json.dumps(request.diagnosis, indent=2)}
        
        Please provide:
        1. Detailed treatment recommendations
        2. Medication plan (if needed)
        3. Lifestyle modifications
        4. Follow-up schedule
        5. Warning signs to watch for
        6. Preventive measures
        7. Dietary recommendations
        8. Exercise recommendations (if appropriate)
        9. Recovery milestones and expectations
        10. When to seek immediate medical attention
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert medical AI system creating personalized treatment plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        treatment_plan = response.choices[0].message.content.strip()
        
        # Save to patient history
        append_to_patient_history(
            request.patient_id,
            'system',
            f"Generated treatment plan based on diagnosis dated {request.diagnosis.get('timestamp', 'N/A')}"
        )
        
        # Generate audio version if needed
        audio_bytes = None
        try:
            tts_language = 'en-GB'
            if request.language in ['hi', 'te']:
                tts_language = request.language
                
            tts = gTTS(text=treatment_plan, lang=tts_language)
            audio_io = io.BytesIO()
            tts.write_to_fp(audio_io)
            audio_io.seek(0)
            audio_bytes = base64.b64encode(audio_io.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Error generating audio for treatment plan: {e}")
        
        return {
            "treatment_plan": treatment_plan,
            "audio": audio_bytes,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating treatment plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class FollowUpRequest(BaseModel):
    patient_id: str
    previous_diagnosis: dict
    previous_treatment_plan: dict
    new_symptoms: List[str]
    current_medications: List[str]
    language: str = 'en-GB'

@app.post("/follow_up")
async def get_follow_up_recommendations(request: FollowUpRequest):
    """Generate follow-up recommendations"""
    try:
        if request.language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Supported languages are: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )
            
        lang_instruction = SUPPORTED_LANGUAGES[request.language]
        
        prompt = f"""
        {lang_instruction}
        
        Review the following follow-up information and provide recommendations:
        
        Previous Diagnosis:
        {json.dumps(request.previous_diagnosis, indent=2)}
        
        Previous Treatment Plan:
        {json.dumps(request.previous_treatment_plan, indent=2)}
        
        New Symptoms:
        {json.dumps(request.new_symptoms, indent=2)}
        
        Current Medications:
        {json.dumps(request.current_medications, indent=2)}
        
        Please provide:
        1. Assessment of progress
        2. Recommendations for treatment adjustments
        3. Additional tests needed (if any)
        4. Modified lifestyle recommendations
        5. Next follow-up timeline
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert medical AI system providing follow-up care recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        follow_up_recommendations = response.choices[0].message.content.strip()
        
        append_to_patient_history(
            request.patient_id,
            'system',
            "Generated follow-up recommendations"
        )
        
        return {
            "recommendations": follow_up_recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating follow-up recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
