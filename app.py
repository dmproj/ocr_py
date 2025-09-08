import gradio as gr
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
import re
import base64
import io
from datetime import datetime

# Load the model
model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def parse_cv_api(image):
    """
    Main API function that parses CV images and returns structured maritime data
    """
    try:
        # Handle base64 image if it's a string
        if isinstance(image, str):
            if image.startswith('data:image'):
                # Remove data URL prefix
                image_data = image.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                # Direct base64
                image_bytes = base64.b64decode(image)
                image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        
        # Generate text from image
        generated_ids = model.generate(
            pixel_values,
            max_length=2048,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        
        # Decode the generated text
        decoded_text = processor.batch_decode(generated_ids.sequences)[0]
        decoded_text = decoded_text.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        decoded_text = re.sub(r"<.*?>", "", decoded_text, count=1).strip()
        
        # Parse and structure the extracted text for maritime CVs
        structured_data = parse_maritime_cv(decoded_text)
        
        # Return both text and JSON
        return [decoded_text, json.dumps(structured_data, indent=2)]
        
    except Exception as e:
        error_response = {
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "personal": {"name": None, "surname": None},
            "professional": {"current_rank": None},
            "documents": {"passport_number": None, "us_visa": False, "stcw_valid": False},
            "certificates": [],
            "education": []
        }
        return ["Error processing image", json.dumps(error_response, indent=2)]

def parse_maritime_cv(text):
    """
    Parse extracted text into structured maritime CV data
    """
    cv_data = {
        "personal": {
            "name": None,
            "surname": None,
            "age": None,
            "date_of_birth": None,
            "sex": None,
            "nationality": None,
            "phone": None,
            "email": None,
            "address": None
        },
        "professional": {
            "current_rank": None,
            "years_in_rank": None,
            "fleet_type": None,
            "vessel_type": None,
            "experience_summary": None
        },
        "documents": {
            "passport_number": None,
            "seaman_book": None,
            "us_visa": False,
            "stcw_valid": False
        },
        "certificates": [],
        "education": []
    }
    
    # Text processing patterns
    lines = text.split('\n')
    text_lower = text.lower()
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match:
        cv_data["personal"]["email"] = email_match.group()
    
    # Extract phone
    phone_pattern = r'[\+]?[1-9]?[\d\s\-\(\)]{8,}'
    phone_matches = re.findall(phone_pattern, text)
    if phone_matches:
        for match in phone_matches:
            clean_match = re.sub(r'[^\d+]', '', match)
            if len(clean_match) >= 8 and not re.match(r'^\d{2}[\.\-/]\d{2}[\.\-/]\d{4}$', match):
                cv_data["personal"]["phone"] = match.strip()
                break
    
    # Enhanced name extraction
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Name extraction patterns
        if any(keyword in line_lower for keyword in ['name', 'surname']):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^[A-Za-z\s\'-]+$', next_line) and len(next_line) > 1:
                    if 'first' in line_lower or 'given' in line_lower:
                        cv_data["personal"]["name"] = next_line
                    elif 'surname' in line_lower or 'last' in line_lower or 'family' in line_lower:
                        cv_data["personal"]["surname"] = next_line
        
        # Age extraction
        if 'age' in line_lower:
            age_match = re.search(r'\b(\d{2})\b', line)
            if age_match:
                age = int(age_match.group(1))
                if 18 <= age <= 70:
                    cv_data["personal"]["age"] = age
        
        # Date of birth
        if any(keyword in line_lower for keyword in ['birth', 'dob', 'born']):
            date_patterns = [
                r'(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{4})',
                r'(\d{4})[\/\.\-](\d{1,2})[\/\.\-](\d{1,2})'
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, line)
                if date_match:
                    cv_data["personal"]["date_of_birth"] = date_match.group()
                    break
        
        # Sex/Gender
        if any(keyword in line_lower for keyword in ['sex', 'gender']):
            if 'male' in line_lower and 'female' not in line_lower:
                cv_data["personal"]["sex"] = "Male"
            elif 'female' in line_lower:
                cv_data["personal"]["sex"] = "Female"
        
        # Nationality
        if 'nationality' in line_lower:
            nationality_match = re.search(r'nationality\s*[:\-]?\s*([A-Za-z\s]+)', line, re.IGNORECASE)
            if nationality_match:
                cv_data["personal"]["nationality"] = nationality_match.group(1).strip()
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^[A-Za-z\s]+$', next_line):
                    cv_data["personal"]["nationality"] = next_line
        
        # Maritime ranks
        maritime_ranks = [
            'master', 'captain', 'chief officer', 'chief mate', 'first officer', 
            'second officer', 'third officer', 'chief engineer', 'first engineer',
            'second engineer', 'third engineer', 'bosun', 'able seaman', 'ordinary seaman',
            'deckhand', 'motorman', 'oiler', 'wiper', 'cook', 'steward', 'electrician',
            'eto', 'pumpman'
        ]
        
        if any(keyword in line_lower for keyword in ['position as', 'rank', 'applying for']):
            if i + 1 < len(lines):
                next_line = lines[i + 1].lower().strip()
                for rank in maritime_ranks:
                    if rank in next_line:
                        cv_data["professional"]["current_rank"] = lines[i + 1].strip()
                        break
        
        # Years in rank
        if any(keyword in line_lower for keyword in ['years', 'experience', 'service']):
            years_match = re.search(r'(\d+)\s*(?:years?|yrs?)', line_lower)
            if years_match:
                cv_data["professional"]["years_in_rank"] = int(years_match.group(1))
        
        # Vessel types
        vessel_types = [
            'tanker', 'container', 'bulk carrier', 'cruise', 'offshore', 'psv', 'ahts',
            'fpso', 'drilling', 'cable laying', 'pipe laying', 'ferry', 'ro-ro',
            'lng', 'lpg', 'oil rig', 'drill ship', 'supply vessel'
        ]
        
        for vessel in vessel_types:
            if vessel in line_lower:
                cv_data["professional"]["vessel_type"] = vessel.title()
                if vessel in ['tanker', 'container', 'bulk carrier', 'lng', 'lpg']:
                    cv_data["professional"]["fleet_type"] = "Merchant"
                elif vessel in ['offshore', 'psv', 'ahts', 'fpso', 'drilling', 'oil rig']:
                    cv_data["professional"]["fleet_type"] = "Offshore"
                elif vessel in ['cruise', 'ferry']:
                    cv_data["professional"]["fleet_type"] = "Cruises-Yachts"
                break
        
        # Certificates
        cert_keywords = ['stcw', 'certificate', 'license', 'endorsement', 'gmdss', 'dp', 'coc']
        if any(keyword in line_lower for keyword in cert_keywords):
            cv_data["certificates"].append(line.strip())
            if 'stcw' in line_lower:
                cv_data["documents"]["stcw_valid"] = True
        
        # Education
        edu_keywords = ['academy', 'institute', 'university', 'college', 'school']
        if any(keyword in line_lower for keyword in edu_keywords):
            cv_data["education"].append({
                "institution": line.strip(),
                "from_year": None,
                "to_year": None
            })
    
    # US Visa detection
    if any(keyword in text_lower for keyword in ['us visa', 'c1/d', 'american visa']):
        cv_data["documents"]["us_visa"] = True
    
    # Extract passport/seaman book numbers
    passport_patterns = [
        r'passport\s*[:\-]?\s*([A-Z0-9]{6,12})',
        r'passport\s*(?:number|no)?\s*[:\-]?\s*([A-Z0-9]{6,12})'
    ]
    
    for pattern in passport_patterns:
        passport_match = re.search(pattern, text, re.IGNORECASE)
        if passport_match:
            cv_data["documents"]["passport_number"] = passport_match.group(1)
            break
    
    seaman_patterns = [
        r'seaman\s*(?:book|id)?\s*[:\-]?\s*([A-Z0-9]{6,12})',
        r'seaman\s*(?:book|id)?\s*(?:number|no)?\s*[:\-]?\s*([A-Z0-9]{6,12})'
    ]
    
    for pattern in seaman_patterns:
        seaman_match = re.search(pattern, text, re.IGNORECASE)
        if seaman_match:
            cv_data["documents"]["seaman_book"] = seaman_match.group(1)
            break
    
    return cv_data

# Create the interface
demo = gr.Interface(
    fn=parse_cv_api,
    inputs=gr.Image(type="pil", label="Upload CV/Resume Image"),
    outputs=[
        gr.Textbox(label="Extracted Text", lines=10),
        gr.Textbox(label="Structured JSON Data", lines=20)
    ],
    title="Maritime CV Parser",
    description="Upload a CV/resume image to extract structured JSON data for maritime professionals.",
    flagging_mode="never",
    analytics_enabled=False
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )
    
