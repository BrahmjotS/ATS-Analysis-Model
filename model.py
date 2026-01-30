import torch
import json
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaModelLoader:
    """Load and manage the fine-tuned Gemma model with PEFT adapters."""
    
    def __init__(self, model_path: str = "gemma_tuned"):
        self.model_path = model_path
        self.device = self._detect_device()
        self.tokenizer = None
        self.model = None
        self.system_prompt = self._load_system_prompt()
        
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _detect_device(self) -> str:
        """Detect and return available device (cuda/cpu)."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"✓ CUDA available - Using NVIDIA GPU")
            logger.info(f"  GPU Name: {gpu_name}")
            logger.info(f"  GPU Memory: {gpu_memory:.2f} GB")
            logger.info(f"  Intel integrated graphics will NOT be used")
        else:
            device = "cpu"
            logger.info("CUDA not available. Using CPU (Intel integrated graphics may be used for display only).")
        return device
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        prompt_path = os.path.join("prompts", "recruiter_system.txt")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Could not load system prompt: {e}")
            return "You are a senior ATS recruiter. Analyze resumes and return structured JSON."
    
    def _load_model(self):
        """Load the base model and apply PEFT adapters if available."""
        try:
            # Check if adapter files exist
            adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
            adapter_model_path = os.path.join(self.model_path, "adapter_model.safetensors")
            
            # Determine base model name (typically gemma-2b or gemma-7b)
            # Try to infer from adapter config or use default
            base_model_name = "google/gemma-2-2b-it"  # Default, adjust if needed
            
            if os.path.exists(adapter_config_path):
                try:
                    with open(adapter_config_path, "r") as f:
                        adapter_config = json.load(f)
                        if "base_model_name_or_path" in adapter_config:
                            base_model_name = adapter_config["base_model_name_or_path"]
                except (json.JSONDecodeError, KeyError, IOError) as e:
                    logger.warning(f"Could not read adapter config: {e}")
            
            logger.info(f"Loading base model: {base_model_name}")
            
            # Load tokenizer (try from model path first, then base model)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                logger.info("Loaded tokenizer from model path")
            except (OSError, ValueError) as e:
                logger.info(f"Could not load tokenizer from model path: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    trust_remote_code=True
                )
                logger.info("Loaded tokenizer from base model")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            if self.device == "cuda":
                logger.info("Loading model on GPU with CUDA...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    offload_folder="offload",
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info(f"Base model loaded on device: {next(base_model.parameters()).device}")
            else:
                logger.info("Loading model on CPU...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True
                )
                base_model = base_model.to(self.device)
            
            # Load PEFT adapters if they exist
            if os.path.exists(adapter_model_path) or os.path.exists(adapter_config_path):
                logger.info("Loading PEFT adapter weights...")
                if self.device == "cuda":
                    self.model = PeftModel.from_pretrained(
                        base_model,
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        offload_folder="offload"
                    )
                    # Verify model is on GPU
                    model_device = next(self.model.parameters()).device
                    logger.info(f"PEFT adapter loaded on device: {model_device}")
                    # Accelerate handles device placement, so manual move is not needed and errors

                else:
                    self.model = PeftModel.from_pretrained(
                        base_model,
                        self.model_path,
                        torch_dtype=torch.float32
                    )
                    self.model = self.model.to(self.device)
                logger.info("PEFT adapter loaded successfully")
            else:
                logger.warning("No adapter weights found. Using base model.")
                self.model = base_model
                # Accelerate handles device placement, so manual move is not needed and errors

            
            self.model.eval()
            # Final device verification
            final_device = next(self.model.parameters()).device
            logger.info(f"Model loaded successfully on device: {final_device}")
            if self.device == "cuda" and final_device.type == "cuda":
                logger.info(f"✓ GPU acceleration enabled on {torch.cuda.get_device_name(0)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_analysis(self, resume_text: str, violations: str) -> Dict:
        """Generate resume analysis using the model."""
        try:
            # Truncate resume text if too long to speed up processing
            # Keep system prompt and violations, but limit resume text
            max_resume_length = 2000  # Limit resume text to ~2000 chars
            original_length = len(resume_text)
            if original_length > max_resume_length:
                resume_text = resume_text[:max_resume_length] + "\n[... Resume text truncated for faster processing ...]"
                logger.info(f"Resume text truncated from {original_length} to {max_resume_length} characters for faster processing")
            
            # Construct prompt
            prompt = f"""{self.system_prompt}

{violations}

RESUME TEXT:
{resume_text}

Analyze this resume and return ONLY a valid JSON object matching this exact schema. 
IMPORTANT: 
1. Use double quotes for all keys and string values.
2. Do not include markdown code blocks (```json ... ```).
3. Do not include trailing commas.
4. Ensure all boolean-like values are strings ("Yes"/"No").

Schema:
{{
    "overall_score": <integer 0-100>,
    "ats_friendly": "<Yes|Intermediate|No>",
    "strengths": "<string description>",
    "x_factor": "<string description>",
    "weaknesses": "<string description>",
    "fixes": ["<string>", ...],
    "suggested_roles": ["<string>", ...],
    "ats_keywords_to_add": ["<string>", ...]
}}

Return ONLY the JSON object string:"""
            
            # Tokenize
            tokenize_start = time.time()
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(next(self.model.parameters()).device)
            tokenize_time = time.time() - tokenize_start
            logger.info(f"Tokenization took {tokenize_time:.2f}s")
            
            # Generate with optimized settings for faster inference
            generate_start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,  # Lower temperature for more deterministic JSON
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=True
                )
            generate_time = time.time() - generate_start
            logger.info(f"Generation took {generate_time:.2f}s, generated {len(outputs[0]) - len(inputs['input_ids'][0])} tokens")
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the newly generated text (remove input prompt)
            if len(inputs['input_ids'][0]) > 0:
                input_length = len(inputs['input_ids'][0])
                generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            # Clean up potential markdown formatting
            generated_text = generated_text.replace("```json", "").replace("```", "").strip()

            # Extract JSON from response
            json_start = generated_text.find("{")
            json_end = generated_text.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                try:
                    # First try standard parsing
                    analysis = json.loads(json_str)
                    return analysis
                except json.JSONDecodeError:
                    # Attempt repairs
                    import re
                    
                    logger.warning("Standard JSON parse failed, attempting repairs...")
                    
                    # 1. Replace single quotes wrapping keys/values with double quotes
                    # This regex finds 'key': or 'value' patterns and replaces ' with "
                    # It's a heuristic and might affect apostrophes in text, but better than failure
                    
                    # Fix keys: 'key': -> "key":
                    json_str_repaired = re.sub(r"'([^']+)'\s*:", r'"\1":', json_str)
                    
                    # Fix string values: : 'value' -> : "value"
                    json_str_repaired = re.sub(r":\s*'([^']*)'", r': "\1"', json_str_repaired)
                    
                    # Fix arrays: ['a', 'b'] -> ["a", "b"]
                    # This is harder to do safely with regex alone for nested structures, 
                    # so we fall back to a simpler global replace if the specific ones fail
                    
                    try:
                        analysis = json.loads(json_str_repaired)
                        return analysis
                    except json.JSONDecodeError:
                        # Fallback: aggressive single quote replacement
                        # We use a custom logic to preserve internal apostrophes if possible
                        # But for now, simple replacement is the vital fix for the reported error
                        json_str_aggr = json_str.replace("'", '"')
                        try:
                            analysis = json.loads(json_str_aggr)
                            return analysis
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON after repairs: {e}")
                            logger.debug(f"Repaired JSON: {json_str_repaired[:500]}")
                            return self._default_response()
            else:
                logger.warning("Could not find JSON in model response")
                return self._default_response()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            if 'generated_text' in locals() and 'json_start' in locals():
                logger.error(f"Generated text snippet: {generated_text[json_start:json_end+100] if json_start >= 0 else 'N/A'}")
            return self._default_response()
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory: {e}")
            logger.error("Try reducing max_new_tokens or using CPU")
            return self._default_response()
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return self._default_response()
    
    def _default_response(self) -> Dict:
        """Return default response if generation fails."""
        return {
            "overall_score": 50,
            "ats_friendly": "Intermediate",
            "strengths": "Unable to generate analysis. Please check model and try again.",
            "x_factor": "N/A",
            "weaknesses": "Model generation error occurred.",
            "fixes": ["Review resume formatting", "Check ATS compatibility"],
            "suggested_roles": ["General Position"],
            "ats_keywords_to_add": ["skills", "experience"]
        }

