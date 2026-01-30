import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from models import ResumeAnalysis
from validators import ResumeValidator
from model import GemmaModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Initialize model loader (global)
model_loader = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model_loader():
    """Get or initialize the model loader."""
    global model_loader
    if model_loader is None:
        try:
            model_loader = GemmaModelLoader(model_path="gemma_tuned")
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    return model_loader

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        model_available = model_loader is not None
        return jsonify({
            'status': 'ok',
            'model_loaded': model_available
        })
    except:
        return jsonify({
            'status': 'ok',
            'model_loaded': False
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis."""
    try:
        model_loader = get_model_loader()
    except Exception as e:
        logger.error(f"Model loader error: {e}")
        return jsonify({'error': 'Model not available. Please check server logs.'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF and DOCX are allowed.'}), 400
    
    filepath = None
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing file: {filename}")
        
        # Initialize validator
        validator = ResumeValidator()
        
        # Extract text and validate based on file type
        if filename.endswith('.pdf'):
            resume_text, metadata = validator.validate_pdf(filepath)
        elif filename.endswith('.docx'):
            resume_text, metadata = validator.validate_docx(filepath)
        else:
            # Clean up file before returning error
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Check if text was extracted
        if not resume_text or len(resume_text.strip()) < 10:
            # Clean up file before returning error
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return jsonify({'error': 'Could not extract text from file. Please ensure the file contains readable text.'}), 400
        
        # Format violations for model context
        violations_text = validator.format_violations_for_prompt(metadata['violations'])
        
        # Generate analysis using model
        if model_loader is None:
            # Clean up file before returning error
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return jsonify({'error': 'Model not initialized'}), 500
        
        analysis_dict = model_loader.generate_analysis(resume_text, violations_text)
        
        # Validate with Pydantic
        try:
            analysis = ResumeAnalysis(**analysis_dict)
        except Exception as e:
            logger.error(f"Pydantic validation error: {e}")
            # Clean up file before returning error
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return jsonify({'error': f'Invalid model output: {str(e)}'}), 500
        
        # Clean up uploaded file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                logger.warning(f"Could not remove uploaded file: {e}")
        
        # Return analysis
        return jsonify({
            'analysis': analysis.model_dump(),
            'metadata': {
                'page_count': metadata['page_count'],
                'fonts': metadata['fonts'],
                'links': metadata['links'],
                'violations': metadata['violations']
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        # Clean up file on any error
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/result')
def result():
    """Render the result page."""
    return render_template('result.html')

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with JSON response."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors with JSON response."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error. Please try again later.'}), 500

@app.errorhandler(413)
def request_too_large(error):
    """Handle file too large errors with JSON response."""
    return jsonify({'error': 'File size exceeds 16MB limit'}), 413

if __name__ == '__main__':
    # Initialize model before starting server
    try:
        model_loader = get_model_loader()
        logger.info("Model loaded. Starting Flask server...")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Server will start but model operations will fail.")
    
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

