from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import shutil
import uuid
import time
import re
import torch
import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel, Field
import uvicorn
import boto3 
import random


app = FastAPI(
    title="Insurance PDF Processor with Clause Matching",
    description="API for processing insurance PDFs and matching clauses",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Angular app's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
bedrock_client = None

def initialize_bedrock_client():
    """Initialize AWS Bedrock Runtime client"""
    global bedrock_client
    if bedrock_client is None:
        try:
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name='ap-south-1'  # Your region
            )
            print(f"AWS Bedrock client initialized successfully")
        except Exception as e:
            print(f"Error initializing Bedrock client: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Bedrock client: {str(e)}")
    return bedrock_client

def get_bedrock_client():
    """Get the initialized Bedrock client"""
    global bedrock_client
    if bedrock_client is None:
        initialize_bedrock_client()
    return bedrock_client

# Import necessary functions from the provided code
from pdf_pro import (
    extract_structured_text,
    create_structural_chunks,
    add_overlap_to_chunks,
    store_chunks_in_vector_db,
    retrieve_relevant_chunks,
    assemble_context,
    generate_response_with_bedrock,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE_TARGET,
    CHUNK_OVERLAP
)

# Configuration
UPLOAD_DIR = "./uploads"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "insurance_policy"
MODEL_PATH = "/home/yash/Desktop/chat/new_model/gemma-3-12b-it-UD-Q8_K_XL.gguf/Unsloth.Q8_K_XL.gguf"
RESULTS_DIR = "./results"

# Bedrock Configuration
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"  # or whichever Claude model you want to use

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Request model for PDF upload with questions
class PDFUploadWithQuestionsRequest(BaseModel):
    questions: List[str]

# Initialize global variables and models for embedding
embedding_model = None
# Store reference clauses at app startup for better performance
reference_clauses = []

# Common insurance terms for normalization and matching
coverage_terms = [
    "fee", "cost", "expense", "removal", "damage", "loss", "clause",
    "minimization", "prevention", "preparation", "repair", "demolition",
    "reinstatement", "extension", "cover", "property", "material"
]

# Load embedding model 
def load_models():
    global embedding_model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Move embedding model to GPU if available
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        embedding_model = embedding_model.to('cuda')
        print("Embedding model moved to GPU")
    else:
        print("WARNING: CUDA is not available for embeddings, using CPU")

# Delete vector DB collection
def clear_vector_db():
    try:
        from chromadb import PersistentClient
        client = PersistentClient(path=DB_PATH)
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"Deleted collection {COLLECTION_NAME}")
        except Exception as e:
            print(f"No collection to delete or error: {e}")
    except Exception as e:
        print(f"Error clearing vector DB: {e}")

# Process PDF and create vector DB
def process_pdf(pdf_path: str):
    try:
        # Clear existing collection
        clear_vector_db()
        
        # Extract structured text
        print("Extracting structured text from PDF...")
        structured_blocks = extract_structured_text(pdf_path)
        print(f"Extracted {len(structured_blocks)} text blocks.")
        
        # Create structural chunks
        print("Creating structural chunks...")
        structural_chunks = create_structural_chunks(structured_blocks)
        print(f"Created {len(structural_chunks)} structural chunks.")
        
        # Add overlap to chunks
        print("Adding overlap to chunks...")
        overlapping_chunks = add_overlap_to_chunks(structural_chunks, overlap=CHUNK_OVERLAP)
        print(f"Processed {len(overlapping_chunks)} chunks with overlap.")
        
        # Store chunks in vector DB
        print("Storing chunks in vector database...")
        embedding_model = store_chunks_in_vector_db(
            overlapping_chunks,
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            model_name=EMBEDDING_MODEL_NAME
        )
        print("Vector database created successfully.")
        
        return embedding_model
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise e

# Process a single question
def process_question(question: str, embedding_model):
    try:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(
            question,
            embedding_model,
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            top_k=15,
            rerank=True
        )
        
        # Assemble context
        context = assemble_context(question, relevant_chunks)
        
        # Generate response - pass the bedrock client
        bedrock_client = get_bedrock_client()
        answer = generate_response_with_bedrock(question, context, bedrock_client)
        
        return {
            "question": question,
            "answer": answer
        }
    except Exception as e:
        print(f"Error processing question: {e}")
        return {
            "question": question,
            "answer": f"Error processing this question: {str(e)}"
        }

# Parse clauses from JSON data
def parse_json_clauses(json_data: dict) -> List[str]:
    """Parse policy clauses from JSON data"""
    
    clauses = []
    
    # Handle different possible JSON structures
    if isinstance(json_data, list):
        clauses = [str(item).strip() for item in json_data if item]
    elif isinstance(json_data, dict):
        # Look for a "Data" field first (common structure from frontend)
        if "Data" in json_data and isinstance(json_data["Data"], list):
            clauses = [str(item).strip() for item in json_data["Data"] if item]
        else:
            # Try to extract clauses from various potential structures
            for key, value in json_data.items():
                if isinstance(value, list):
                    clauses.extend([str(item).strip() for item in value if item])
                elif isinstance(value, str):
                    clauses.append(value.strip())
                elif isinstance(value, dict):
                    # Handle nested dicts
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, list):
                            clauses.extend([str(item).strip() for item in subvalue if item])
                        elif isinstance(subvalue, str):
                            clauses.append(subvalue.strip())
    
    # Filter out very short items that are likely not complete clauses
    clauses = [clause for clause in clauses if len(clause) > 3]
    
    # Process clauses to handle common formatting
    expanded_clauses = []
    for clause in clauses:
        if " - " in clause or " – " in clause:
            parts = re.split(r' - | – ', clause)
            if all(len(part) > 3 for part in parts):
                expanded_clauses.extend(parts)
            else:
                expanded_clauses.append(clause)
        else:
            expanded_clauses.append(clause)
    
    # Remove duplicates while preserving order
    clean_clauses = []
    seen = set()
    for clause in expanded_clauses:
        cleaned = clause.strip()
        if cleaned and cleaned not in seen and len(cleaned) >= 3:
            clean_clauses.append(cleaned)
            seen.add(cleaned)
    
    return clean_clauses

# Normalize text for better matching of insurance clauses
def normalize_text(text: str) -> str:
    """Normalize text for better matching of insurance clauses"""
    # Convert to lowercase
    text = text.lower()
    
    # Replace common insurance abbreviations and symbols
    replacements = {
        "incl.": "including",
        "excl.": "excluding",
        "&": "and",
        "w/": "with",
        "w/o": "without",
        "%": "percent",
        "/": " ",  # Replace slashes with spaces to handle "fire/water" as "fire water"
        "–": "-",  # Standardize dashes
        "—": "-",
    }
    for abbr, full in replacements.items():
        text = text.replace(abbr, full)
    
    # Handle parenthetical qualifiers common in insurance
    text = re.sub(r'\([^)]*\)', ' ', text)
    
    # Handle common insurance terminology variations
    text = text.replace("fire brigade", "firefighting")
    text = text.replace("fire fighting", "firefighting")
    text = text.replace("shut down", "shutdown")
    text = text.replace("start up", "startup")
    
    # Remove common filler words
    text = re.sub(r'\b(the|a|an|of|for|to|in|on|by|this|that|all|any)\b', ' ', text)
    
    # Replace plurals with singular forms for key insurance terms
    for term in coverage_terms:
        text = re.sub(rf'\b{term}s\b', term, text)
    
    # Remove remaining punctuation except hyphens (important in insurance terms)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Calculate string similarity using SequenceMatcher
def string_similarity(str1: str, str2: str) -> float:
    """Calculate string similarity using SequenceMatcher"""
    return SequenceMatcher(None, str1, str2).ratio()

# Find exact and fuzzy matches between clauses
def find_matches(json_clauses: List[str], pdf_clauses: List[str]) -> List[Tuple[str, str, float]]:
    """Find exact and fuzzy matches between clauses"""
    matches = []
    
    # Create tracking sets for matched items
    matched_pdf = set()
    matched_json = set()
    
    # Step 1: Create normalized versions of all clauses for faster matching
    norm_json_map = {normalize_text(item): item for item in json_clauses}
    norm_pdf_map = {normalize_text(item): item for item in pdf_clauses}
    
    # Step 2: Find exact normalized matches
    for norm_json, orig_json in norm_json_map.items():
        if orig_json in matched_json:
            continue
            
        for norm_pdf, orig_pdf in norm_pdf_map.items():
            if orig_pdf in matched_pdf:
                continue
            
            # Exact match on normalized text
            if norm_json == norm_pdf and len(norm_json) > 5:  # Avoid very short matches
                matches.append((orig_json, orig_pdf, 1.0))  # Score 1.0 for exact match
                matched_pdf.add(orig_pdf)
                matched_json.add(orig_json)
    
    # Step 3: Find containment matches (one string fully contains the other)
    for norm_json, orig_json in norm_json_map.items():
        if orig_json in matched_json:
            continue
            
        for norm_pdf, orig_pdf in norm_pdf_map.items():
            if orig_pdf in matched_pdf:
                continue
            
            # Check if one string fully contains the other (after normalization)
            if (norm_json in norm_pdf or norm_pdf in norm_json) and min(len(norm_json), len(norm_pdf)) > 5:
                # Calculate length ratio to ensure meaningful containment
                length_ratio = min(len(norm_json), len(norm_pdf)) / max(len(norm_json), len(norm_pdf))
                
                # For shorter strings, require higher ratio
                ratio_threshold = 0.5  # Lower this for more matches
                
                if length_ratio > ratio_threshold:
                    matches.append((orig_json, orig_pdf, 0.95))  # High score for containment
                    matched_pdf.add(orig_pdf)
                    matched_json.add(orig_json)
    
    # Step 4: Find fuzzy matches for remaining items
    for norm_json, orig_json in norm_json_map.items():
        if orig_json in matched_json:
            continue
            
        for norm_pdf, orig_pdf in norm_pdf_map.items():
            if orig_pdf in matched_pdf:
                continue
            
            # Skip very short strings
            if len(norm_json) < 5 or len(norm_pdf) < 5:
                continue
            
            # Calculate similarity
            similarity = string_similarity(norm_json, norm_pdf)
            
            # Check if it meets our threshold (0.7 for fuzzy matches - lower threshold)
            if similarity >= 0.7:
                matches.append((orig_json, orig_pdf, similarity))
                matched_pdf.add(orig_pdf)
                matched_json.add(orig_json)
    
    # Sort by similarity score
    matches.sort(key=lambda x: x[2], reverse=True)
    
    return matches

# Find semantic matches using embeddings
def find_semantic_matches(json_clauses: List[str], pdf_clauses: List[str], 
                         exact_fuzzy_matches: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
    """Find semantic matches using embeddings"""
    global embedding_model
    
    # Create sets of already matched clauses
    matched_json = set(j for j, _, _ in exact_fuzzy_matches)
    matched_pdf = set(p for _, p, _ in exact_fuzzy_matches)
    
    # Get remaining unmatched clauses
    remaining_json = [j for j in json_clauses if j not in matched_json]
    remaining_pdf = [p for p in pdf_clauses if p not in matched_pdf]
    
    if not remaining_json or not remaining_pdf:
        return []
    
    semantic_matches = []
    try:
        # Ensure the embedding model is loaded
        if embedding_model is None:
            load_models()
            
        # Use GPU for embeddings if available
        with torch.no_grad():  # Save memory
            # Compute embeddings
            json_embeddings = embedding_model.encode(remaining_json, convert_to_tensor=True)
            pdf_embeddings = embedding_model.encode(remaining_pdf, convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_scores = util.pytorch_cos_sim(json_embeddings, pdf_embeddings)
        
        # Find pairs with high semantic similarity
        for i, json_item in enumerate(remaining_json):
            # Find best match for each JSON clause
            best_match_idx = torch.argmax(cosine_scores[i]).item()
            best_match_score = cosine_scores[i][best_match_idx].item()
            
            # Use a threshold for matching
            if best_match_score > 0.6:
                pdf_item = remaining_pdf[best_match_idx]
                semantic_matches.append((json_item, pdf_item, best_match_score))
        
    except Exception as e:
        print(f"Error during semantic matching: {e}")
    
    return semantic_matches

# Compare clauses and find matches
def compare_clauses(reference_clauses: List[str], user_clauses: List[str]) -> Dict[str, Any]:
    """Compare clauses and find matches"""
    
    if not reference_clauses or not user_clauses:
        return {
            "status": "error",
            "message": "Missing clause data",
            "matches": [],
            "semantic_matches": []
        }
    
    # Find exact and fuzzy matches
    matches = find_matches(reference_clauses, user_clauses)
    
    # Find semantic matches using embeddings
    semantic_matches = find_semantic_matches(reference_clauses, user_clauses, matches)
    
    # Prepare results
    results = {
        "status": "success",
        "message": f"Found {len(matches)} direct matches and {len(semantic_matches)} semantic matches",
        "matches": [{"reference": j, "user": p, "score": s} for j, p, s in matches],
        "semantic_matches": [{"reference": j, "user": p, "score": s} for j, p, s in semantic_matches]
    }
    
    return results

# Extract clauses from LLM response
def extract_clauses_from_llm_response(llm_response: str) -> List[str]:
    """Extract individual clauses from LLM response"""
    clauses = []
    
    # Split by common separators (newlines, bullets, numbers)
    # This is a simplified approach - expand based on actual LLM outputs
    lines = llm_response.split('\n')
    
    for line in lines:
        # Remove leading numbers, bullets, etc.
        cleaned_line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
        
        # Skip empty lines
        if not cleaned_line:
            continue
            
        # Add as clause if it's substantial enough
        if len(cleaned_line) > 5:
            clauses.append(cleaned_line)
    
    return clauses

@app.on_event("startup")
async def startup_event():
    """Load models and reference data when the application starts"""
    global reference_clauses
    
    # Initialize Bedrock client first
    try:
        initialize_bedrock_client()
        print("✓ Bedrock client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Bedrock client: {e}")
        # You might want to decide if you want to continue without Bedrock or exit
    
    # Load embedding model
    try:
        load_models()
        print("✓ Embedding models loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load embedding models: {e}")
    
    # Load reference clauses from file
    try:
        with open('covers.json', 'r', encoding='utf-8') as f:
            reference_data = json.load(f)
            reference_clauses = parse_json_clauses(reference_data)
            print(f"✓ Loaded {len(reference_clauses)} reference policy clauses")
    except Exception as e:
        print(f"⚠ WARNING: Failed to load reference clauses: {str(e)}")
        # Set a default list of clauses in case the file is missing
        reference_clauses = [
            "Professional Fees - Costs for professional services",
            "Debris Removal - Costs to remove debris following an insured event",
            "Fire Brigade Charges - Costs charged by authorities",
            "Capital Additions - Newly acquired property"
        ]
        print(f"✓ Using default reference clauses ({len(reference_clauses)} items)")
    
    print("🚀 Insurance PDF Processor API is ready")

def parse_sum_insured_to_structured_json(answer_text, filter_zero_values=True):
    """
    Parse the sum insured answer text and convert it to structured JSON format
    Returns format with sections and locations structure as specified
    
    Args:
        answer_text (str): The raw answer text from LLM
        filter_zero_values (bool): If True, excludes items with amount = 0
    """
    try:
        import re
        sections_dict = {}
        current_location = None
        current_section = None
        current_items = []
        
        lines = answer_text.strip().split('\n')
        
        def save_current_data():
            """Helper function to save current location/section/items data"""
            if current_location and current_section and current_items:
                if current_section not in sections_dict:
                    sections_dict[current_section] = []
                
                # Filter out zero values if requested
                filtered_items = current_items
                if filter_zero_values:
                    filtered_items = [item for item in current_items if item["amount"] > 0]
                
                # Only save if there are items after filtering
                if filtered_items:
                    # Check if location already exists in this section
                    location_exists = False
                    for existing_loc in sections_dict[current_section]:
                        if existing_loc["location"] == current_location:
                            existing_loc["items"].extend(filtered_items)
                            location_exists = True
                            break
                    
                    if not location_exists:
                        sections_dict[current_section].append({
                            "location": current_location,
                            "items": filtered_items.copy()
                        })
        
        # Check if we have a location without section header - add default section
        has_section_header = any(re.match(r'^\s*section(\s+name)?\s*:', line, re.IGNORECASE) for line in lines)
        has_location_header = any(re.match(r'^\s*(client\s+)?location\s*:', line, re.IGNORECASE) for line in lines)
        has_block_header = any(re.match(r'^\s*name\s+of\s+block\s*:', line, re.IGNORECASE) for line in lines)
        
        # If we have location/block but no section, inject default section
        if (has_location_header or has_block_header) and not has_section_header:
            # Find the first location/block line and insert section before it
            for i, line in enumerate(lines):
                if re.match(r'^\s*(client\s+)?location\s*:', line, re.IGNORECASE) or re.match(r'^\s*name\s+of\s+block\s*:', line, re.IGNORECASE):
                    lines.insert(i, "Section: Property Damage")
                    break
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip "Sum Insured (`)" header line
            if re.match(r'^\s*sum\s+insured\s*\([^)]*\)\s*$', line, re.IGNORECASE):
                continue
            
            # Skip "Total Sum Insured" lines
            if re.match(r'^\s*total\s+sum\s+insured\s*:', line, re.IGNORECASE):
                continue
            
            # Check if this line is a location header (case-insensitive, flexible matching)
            # Handles "Location:", "Client Location:", and "Name of Block:" formats
            if re.match(r'^\s*(client\s+)?location\s*:', line, re.IGNORECASE) or re.match(r'^\s*name\s+of\s+block\s*:', line, re.IGNORECASE):
                # Save previous data if exists
                save_current_data()
                
                # Extract location name and reset items
                current_location = line.split(':', 1)[1].strip()
                current_items = []
                
                # If no section is set yet, use default
                if current_section is None:
                    current_section = "Property Damage"
                continue
            
            # Check if this line is a section header (case-insensitive, flexible matching)
            elif re.match(r'^\s*section(\s+name)?\s*:', line, re.IGNORECASE):
                # Don't save previous data here since we're still building the same location
                # Extract section name
                current_section = line.split(':', 1)[1].strip()
                continue
            
            # Parse item: amount lines
            elif ':' in line and current_location:
                # If we have items but no section set, create default
                if current_section is None:
                    current_section = "Property Damage"
                
                parts = line.split(':', 1)
                if len(parts) == 2:
                    description = parts[0].strip()
                    value_text = parts[1].strip()
                    
                    # Extract numeric amount - more robust parsing
                    amount = 0
                    if value_text.isdigit():
                        amount = int(value_text)
                    elif any(char.isdigit() for char in value_text):
                        # Handle currency symbols, commas, spaces, etc.
                        numeric_match = re.search(r'[\d,]+', value_text.replace(' ', ''))
                        if numeric_match:
                            amount_clean = numeric_match.group().replace(',', '')
                            if amount_clean.isdigit():
                                amount = int(amount_clean)
                    
                    current_items.append({
                        "description": description,
                        "amount": amount
                    })
        
        # Save any remaining data after processing all lines
        save_current_data()
        
        # Convert to required array format
        result_array = []
        for section_name, locations in sections_dict.items():
            result_array.append({
                "section": section_name,
                "locations": locations
            })
        
        # If no data was parsed, return empty structure
        if not result_array:
            return [{"section": "NO_DATA", "locations": [{"location": "N/A", "items": [{"description": "Unable to parse data", "amount": 0}]}]}]
        
        return result_array
        
    except Exception as e:
        print(f"Error parsing sum insured answer: {e}")
        import traceback
        traceback.print_exc()
        # Return error structure in the expected format
        return [{"section": "ERROR", "locations": [{"location": "N/A", "items": [{"description": f"Parse error: {str(e)}", "amount": 0}]}]}]
    

@app.post("/api/upload_pdf_with_questions")
async def upload_pdf_with_questions(
    file: UploadFile = File(...)
):
    # Use default questions always
    default_questions = [
        "list Name of the financiers?",
        "list the addon covers opted?",
        "from block 1 just give name of block and sum insured breakdown?",
        "if terrorism is not covered then just give NO okay?",
        "occupancy description?",
        "just give the insured name?",
        "give me the name of the product?"
    ]
    
    questions_data = PDFUploadWithQuestionsRequest(questions=default_questions)
    try:
        # Validate file is a PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Create a unique filename
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF and get embedding model
        print(f"Processing PDF: {file.filename}")
        pdf_embedding_model = process_pdf(file_path)
        
        # Process each question
        start_time = time.time()
        results = []
        
        for question in questions_data.questions:
            print(f"Processing question: {question}")
            result = process_question(question, pdf_embedding_model)
            results.append(result)
        
        # Save original LLM results to JSON file
        timestamp = int(time.time())
        base_filename = os.path.splitext(file.filename)[0]
        results_filename = f"{base_filename}_results_{timestamp}.json"
        results_path = os.path.join(RESULTS_DIR, results_filename)
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Extract the second answer (addon covers) for matching
        addon_covers_response = results[1]["answer"] if len(results) > 1 else ""
        
        # Extract clauses from the LLM response
        extracted_clauses = extract_clauses_from_llm_response(addon_covers_response)
        
        # Compare with reference clauses
        global reference_clauses
        matching_results = compare_clauses(reference_clauses, extracted_clauses)
        
        # Process sum insured response (third question)
        sum_insured_response = results[2]["answer"] if len(results) > 2 else ""
        sum_insured_structured = parse_sum_insured_to_structured_json(sum_insured_response)
        
        # Build the final JSON structure with all responses
        final_result = {
            "status": "success",
            "pdf_name": file.filename,
            "timestamp": timestamp,
            "responses": {
                "financiers": {
                    "question": default_questions[0],
                    "answer": results[0]["answer"] if results else ""
                },
                "addon_covers": {
                    "question": default_questions[1],
                    "answer": addon_covers_response,
                    "matched_covers": {
                        "matches": matching_results["matches"],
                        "semantic_matches": matching_results["semantic_matches"],
                        "match_status": matching_results["status"] == "success" and (
                            len(matching_results["matches"]) > 0 or 
                            len(matching_results["semantic_matches"]) > 0
                        )
                    }
                },
                "sumInsuredSplit": {
                    "question": default_questions[2],
                    "raw_answer": sum_insured_response,
                    "answer": sum_insured_structured
                },
                "terrorismCover": {
                    "question": default_questions[3],
                    "answer": results[3]["answer"] if len(results) > 3 else ""
                },
                "occupancy_description": {
                    "question": default_questions[4],
                    "answer": results[4]["answer"] if len(results) > 4 else ""
                },
                "client_name": {
                    "question": default_questions[5],
                    "answer": results[5]["answer"] if len(results) > 5 else ""
                },
                "product_name": {
                    "question": default_questions[6],
                    "answer": results[6]["answer"] if len(results) > 6 else ""
                }
            },
            "processing_time": time.time() - start_time
        }
        
        # Save final result with matches
        final_results_filename = f"{base_filename}_final_results_{timestamp}.json"
        final_results_path = os.path.join(RESULTS_DIR, final_results_filename)
        
        with open(final_results_path, "w") as f:
            json.dump(final_result, f, indent=2)
        
        return JSONResponse(final_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF and questions: {str(e)}")
    
# Provide a simple health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Insurance PDF Processor API is running"}

@app.get("/bedrock-status")
async def bedrock_status():
    """Check Bedrock client status"""
    try:
        client = get_bedrock_client()
        return {
            "status": "connected", 
            "message": "Bedrock client is initialized and ready",
            "region": "ap-south-1"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Bedrock client error: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)