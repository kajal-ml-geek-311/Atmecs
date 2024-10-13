# MediAssist AI - Medical Assistant Application

## Overview
MediAssist AI is an AI-powered application designed to assist in medical diagnosis by analyzing medical images, generating synthetic medical reports, and creating personalized treatment plans. It supports multiple languages and provides audio feedback for patients.

## Features
- **Medical Image Diagnosis**: Upload X-ray or other medical images for AI-based analysis.
- **Synthetic Medical Reports**: Automatically generated medical reports based on patient data.
- **Personalized Treatment Plans**: AI-generated, personalized treatment recommendations.
- **Multilingual Support**: Currently supports English and Hindi, with more languages to come.
- **Audio Feedback**: Audio-based diagnoses for enhanced accessibility.

## Technologies Used
- **Backend**: FastAPI, PyTorch, OpenAI, TorchXRayVision, gTTS (Google Text-to-Speech).
- **Frontend**: React.js (or other relevant technologies from the front-end code).
- **Machine Learning Models**: DenseNet for image analysis, OpenAI GPT-4 for natural language generation.

## Setup Instructions

### Backend

#### Prerequisites
- Python 3.8+
- CUDA-capable GPU (if available)

#### Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd backend
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file with your OpenAI API key:
    ```bash
    OPENAI_API_KEY=your_openai_key_here
    ```

5. Run the application:
    ```bash
    uvicorn main:app --reload
    ```

6. Access API documentation at `http://localhost:8000/docs`.

### Frontend

#### Prerequisites
- Node.js and npm

#### Installation

1. Extract the front-end zip file.

2. Install dependencies:
    ```bash
    npm install
    ```

3. Start the front-end server:
    ```bash
    npm start
    ```

### Testing the Application
- Visit `http://localhost:3000` for the front-end.
- Test the endpoints via Swagger UI at `http://localhost:8000/docs`.

## API Endpoints

### `/diagnosis`
- **Method**: POST
- **Description**: Upload a medical image and get an AI-generated diagnosis.
- **Parameters**:
  - `patient_id` (string)
  - `image` (file)
  - `patient_data` (string)
  - `language` (optional, default: 'en')

### `/generate_report`
- **Method**: POST
- **Description**: Generate a synthetic medical report based on patient details.

### `/treatment_plan`
- **Method**: POST
- **Description**: Generate a personalized treatment plan based on patient data.

## Future Enhancements
- Additional language support.
- Expansion to other medical image types.
- Integration with healthcare providers for real-time consultations.

## License
MIT License
