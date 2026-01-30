# ATS Resume Analyzer

A Flask web application that uses a fine-tuned Gemma model to analyze resumes for ATS (Applicant Tracking System) optimization.

## Features

- **AI-Powered Analysis**: Uses fine-tuned Gemma-2-2B model for resume analysis
- **File Support**: Accepts PDF and DOCX resume files
- **Soft-Gated Validation**: Detects formatting violations (fonts, page count, links) without blocking analysis
- **Structured Output**: Returns comprehensive analysis with scores, strengths, weaknesses, and recommendations
- **Modern UI**: Responsive web interface with real-time analysis results
- **Copy-to-Clipboard**: Easy copying of suggested roles and ATS keywords

## Setup

### Option 1: Automated Setup (Recommended)

Run the setup script to automatically create a virtual environment and install dependencies:

```bash
python setup.py
```

Then activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Option 2: Manual Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment (see commands above)

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: For CUDA support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Running the Application

1. Ensure the virtual environment is activated
2. Run the Flask application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
project/
├── prompts/
│   └── recruiter_system.txt    # System prompt for the model
├── templates/
│   ├── index.html              # Upload page
│   └── result.html             # Results display page
├── static/
│   └── style.css               # Styling
├── uploads/                     # Temporary file storage (auto-created)
├── gemma_tuned/                 # Fine-tuned model directory
├── app.py                       # Main Flask application
├── model.py                     # Model loading and inference
├── models.py                    # Pydantic schemas
├── validators.py                # Resume validation logic
├── requirements.txt             # Python dependencies
├── setup.py                     # Setup script
└── README.md                    # This file
```

## API Endpoints

### `GET /`
Main upload page

### `POST /upload`
Upload and analyze a resume file

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (PDF or DOCX)

**Response:**
```json
{
  "analysis": {
    "overall_score": 85,
    "ats_friendly": "Yes",
    "strengths": "...",
    "x_factor": "...",
    "weaknesses": "...",
    "fixes": ["...", "..."],
    "suggested_roles": ["...", "..."],
    "ats_keywords_to_add": ["...", "..."]
  },
  "metadata": {
    "page_count": 1,
    "fonts": ["Calibri"],
    "links": ["mailto:...", "https://..."],
    "violations": []
  }
}
```

## Model Configuration

The application automatically:
- Detects GPU availability and uses CUDA if available
- Loads the fine-tuned Gemma model from `gemma_tuned/`
- Applies PEFT adapter weights if present
- Falls back to CPU if GPU is not available

## Validation Rules

The system performs soft-gated validation (does not block analysis):

1. **Page Count**: Entry-level resumes should be 1 page
2. **Fonts**: Only Sans-Serif or Calibri fonts are acceptable
3. **Links**: Must use `mailto:` for emails and `https://` for web links

Violations are injected into the model context to influence scoring and recommendations.

## Requirements

- Python 3.8+
- PyTorch (with CUDA support optional)
- Flask 3.0+
- Transformers 4.35+
- PEFT 0.6+
- pdfplumber
- python-docx

## Troubleshooting

### Model Loading Issues
- Ensure the `gemma_tuned/` directory contains the adapter files
- Check that you have sufficient RAM/VRAM for the model
- Verify the base model `google/gemma-2-2b-it` is accessible

### CUDA Issues
- Install CUDA-enabled PyTorch if you have a compatible GPU
- The application will automatically fall back to CPU if CUDA is unavailable

### File Upload Issues
- Ensure files are PDF or DOCX format
- Maximum file size is 16MB
- Check that `uploads/` directory is writable
