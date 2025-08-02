from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
import uuid
from typing import Optional, List, Tuple, Dict, Set
import time
import json
import re
import torch
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import boto3


# Import all functions from your rfq.py file
from bed import *

app = FastAPI(
    title="Excel Policy Chat API", 
    description="Upload Excel files and ask questions about insurance policies with semantic matching",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Angular app's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables for models (loaded once for efficiency)
embedding_model = None
semantic_matcher = None
temp_files_dir = "./temp_uploads"

# Create temp directory if it doesn't exist
os.makedirs(temp_files_dir, exist_ok=True)

bedrock_client = None


# Path to covers.json file
COVERS_JSON_PATH = "./covers.json"


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


class InsurancePolicyMatcher:
    def __init__(self):
        """Initialize the matcher with GPU acceleration where possible"""
        print(f"Initializing Insurance Policy Clause Matcher...")
        
        # Load embedding model
        print(f"Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Move embedding model to GPU if available
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            self.embedding_model = self.embedding_model.to('cuda')
            print("Embedding model moved to GPU")
        else:
            print("WARNING: CUDA is not available for embeddings, using CPU")
        
        # Common insurance terms for normalization and matching
        self.coverage_terms = [
            "fee", "cost", "expense", "removal", "damage", "loss", "clause",
            "minimization", "prevention", "preparation", "repair", "sudden","delibrate","demolition",
            "reinstatement", "extension", "cover", "property", "material"
        ]
        
        print("Initialization complete")
    
    def load_json_clauses(self, json_path: str) -> List[str]:
        """Load clauses from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Debug - print raw data
            print(f"JSON Data type: {type(data)}")
            print(f"JSON Data preview: {str(data)[:200]}...")
            
            # Handle different possible JSON structures
            clauses = []
            
            if isinstance(data, list):
                clauses = [str(item).strip() for item in data if item]
            elif isinstance(data, dict):
                # Try to extract clauses from various potential structures
                for key, value in data.items():
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
            
            print(f"Loaded {len(clauses)} clauses from JSON")
            print(f"Sample clauses: {clauses[:3]}")
            return clauses
        except Exception as e:
            print(f"Error loading JSON: {e}")
            print(f"Error details: {str(e)}")
            print(f"Make sure the file exists and contains valid JSON")
            return []
    
    def normalize_text(self, text: str) -> str:
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
        for term in self.coverage_terms:
            text = re.sub(rf'\b{term}s\b', term, text)
        
        # Remove remaining punctuation except hyphens (important in insurance terms)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using SequenceMatcher"""
        return SequenceMatcher(None, str1, str2).ratio()
    
    def find_matches(self, json_clauses: List[str], pdf_clauses: List[str]) -> List[Tuple[str, str, float]]:
        """Find exact and fuzzy matches between JSON and PDF clauses"""
        matches = []
        
        # Create tracking sets for matched items
        matched_pdf = set()
        matched_json = set()
        
        # Step 1: Create normalized versions of all clauses for faster matching
        print("Normalizing clauses for comparison...")
        norm_json_map = {self.normalize_text(item): item for item in json_clauses}
        norm_pdf_map = {self.normalize_text(item): item for item in pdf_clauses}
        
        # Step 2: Find exact normalized matches
        print("Finding exact matches...")
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
                    print(f"Exact match: '{orig_json}' = '{orig_pdf}'")
        
        # Step 3: Find containment matches (one string fully contains the other)
        print("Finding containment matches...")
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
                        print(f"Containment match: '{orig_json}' = '{orig_pdf}'")
        
        # Step 4: Find fuzzy matches for remaining items
        print("Finding fuzzy matches...")
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
                similarity = self.string_similarity(norm_json, norm_pdf)
                
                # Check if it meets our threshold (0.7 for fuzzy matches - lower threshold)
                if similarity >= 0.7:
                    matches.append((orig_json, orig_pdf, similarity))
                    matched_pdf.add(orig_pdf)
                    matched_json.add(orig_json)
                    print(f"Fuzzy match: '{orig_json}' ~ '{orig_pdf}' (score: {similarity:.2f})")
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(matches)} matches")
        return matches
    
    def find_semantic_matches(self, json_clauses: List[str], pdf_clauses: List[str], 
                             exact_fuzzy_matches: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """Find semantic matches using embeddings"""
        # Create sets of already matched clauses
        matched_json = set(j for j, _, _ in exact_fuzzy_matches)
        matched_pdf = set(p for _, p, _ in exact_fuzzy_matches)
        
        # Get remaining unmatched clauses
        remaining_json = [j for j in json_clauses if j not in matched_json]
        remaining_pdf = [p for p in pdf_clauses if p not in matched_pdf]
        
        if not remaining_json or not remaining_pdf:
            print("No remaining unmatched clauses for semantic matching")
            return []
        
        print(f"Computing embeddings for {len(remaining_json)} JSON and {len(remaining_pdf)} PDF clauses...")
        
        semantic_matches = []
        try:
            # Use GPU for embeddings if available
            with torch.no_grad():  # Save memory
                # Compute embeddings
                json_embeddings = self.embedding_model.encode(remaining_json, convert_to_tensor=True)
                pdf_embeddings = self.embedding_model.encode(remaining_pdf, convert_to_tensor=True)
            
            # Compute cosine similarity
            print("Computing cosine similarity...")
            cosine_scores = util.pytorch_cos_sim(json_embeddings, pdf_embeddings)
            
            # Find pairs with high semantic similarity
            for i, json_item in enumerate(remaining_json):
                # Find best match for each JSON clause
                best_match_idx = torch.argmax(cosine_scores[i]).item()
                best_match_score = cosine_scores[i][best_match_idx].item()
                
                # Use a lower threshold for more matches
                if best_match_score > 0.6:
                    pdf_item = remaining_pdf[best_match_idx]
                    semantic_matches.append((json_item, pdf_item, best_match_score))
                    print(f"Semantic match: '{json_item}' ~ '{pdf_item}' (score: {best_match_score:.3f})")
            
        except Exception as e:
            print(f"Error during semantic matching: {e}")
        
        print(f"Found {len(semantic_matches)} semantic matches")
        return semantic_matches
    
    def compare_clauses_with_answers(self, covers_json_path: str, answer_clauses: List[str]) -> Dict:
        """Compare answer clauses with covers.json using the original matching logic"""
        print(f"\nStarting comparison between covers.json and extracted answer clauses")
        
        # Load clauses from covers.json
        covers_clauses = self.load_json_clauses(covers_json_path)
        
        if not covers_clauses:
            print(f"WARNING: No clauses loaded from covers.json: {covers_json_path}")
            return {"error": "Failed to load covers.json clauses", "matches": [], "semantic_matches": [], "unmatched": []}
        
        if not answer_clauses:
            print(f"WARNING: No answer clauses provided")
            return {"error": "No answer clauses provided", "matches": [], "semantic_matches": [], "unmatched": answer_clauses}
        
        # Find exact and fuzzy matches
        print("\nFinding exact and fuzzy matches...")
        exact_fuzzy_matches = self.find_matches(covers_clauses, answer_clauses)
        
        # Find semantic matches using embeddings
        print("\nFinding semantic matches...")
        semantic_matches = self.find_semantic_matches(covers_clauses, answer_clauses, exact_fuzzy_matches)
        
        # Prepare matches in the format needed for response
        matches = []
        semantic_match_list = []
        all_matched_answers = set()  # Changed from all_matched_covers to all_matched_answers
        
        # Add exact/fuzzy matches
        for cover_clause, answer_clause, score in exact_fuzzy_matches:
            match_type = "exact" if score == 1.0 else "fuzzy"
            matches.append({
                "reference": cover_clause,
                "user": answer_clause,
                "type": match_type,
                "score": score
            })
            all_matched_answers.add(answer_clause)  # Track matched answer clauses
        
        # Add semantic matches
        for cover_clause, answer_clause, score in semantic_matches:
            semantic_match_list.append({
                "reference": cover_clause,
                "user": answer_clause,
                "score": score
            })
            all_matched_answers.add(answer_clause)  # Track matched answer clauses
        
        # Find unmatched user/answer clauses (not unmatched covers)
        unmatched_answers = [clause for clause in answer_clauses if clause not in all_matched_answers]
        
        return {
            "matches": matches,
            "semantic_matches": semantic_match_list,
            "unmatched": unmatched_answers  # Now returns unmatched user clauses instead of covers
        }

def initialize_embedding_model():
    """Initialize the embedding model once"""
    global embedding_model
    if embedding_model is None:
        try:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load embedding model: {str(e)}")
    return embedding_model

def initialize_semantic_matcher():
    """Initialize the semantic matcher once"""
    global semantic_matcher
    if semantic_matcher is None:
        try:
            print("Initializing semantic matcher...")
            semantic_matcher = InsurancePolicyMatcher()
            print("Semantic matcher initialized successfully")
        except Exception as e:
            print(f"Error initializing semantic matcher: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize semantic matcher: {str(e)}")
    return semantic_matcher

def extract_clauses_from_answer(answer: str) -> List[str]:
    """Extract clauses from a single answer"""
    clauses = []
    if isinstance(answer, str):
        # Split by common delimiters and clean up
        parts = re.split(r'[,;\n]', answer)
        for part in parts:
            if part.strip() and len(part.strip()) > 3:
                cleaned_part = re.sub(r'^\d+\.\s*', '', part.strip())
                clauses.append(cleaned_part)
    return clauses

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
        
        # If we have location but no section, inject default section
        if has_location_header and not has_section_header:
            # Find the first location line and insert section before it
            for i, line in enumerate(lines):
                if re.match(r'^\s*(client\s+)?location\s*:', line, re.IGNORECASE):
                    lines.insert(i, "Section: Property Damage")
                    break
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a location header (case-insensitive, flexible matching)
            # Handles both "Location:" and "Client Location:" formats
            if re.match(r'^\s*(client\s+)?location\s*:', line, re.IGNORECASE):
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

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Starting up FastAPI application...")
    initialize_embedding_model()
    initialize_semantic_matcher()
    initialize_bedrock_client()  # Add this line
    print("FastAPI application ready!")

@app.post("/api/upload_excel_with_questions")
async def upload_excel_and_query(
    file: UploadFile = File(..., description="Excel file to upload (.xlsx or .xls)")
):
    """
    Upload an Excel file and automatically process it with default questions.
    Automatically answers default questions and compares results with covers.json:
    1. "give me occupancy?"
    2. "give me all the add-on & clauses from fire section?"
    3. "Show location ,section name and finished goods for Mumbai - 422002 in clean text format?"
    4. "Show location ,section name and finished goods for Mumbai - 422003 in clean text format?"
    (Sum insured questions can be easily extended by adding more questions to the sum_insured_questions list)

    - **file**: Excel file (.xlsx or .xls format)

    Returns answers to the default questions and semantic matching results.
    """

    # Validate file type
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an Excel file (.xlsx or .xls)"
        )

    # Generate unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    temp_filename = f"{unique_id}{file_extension}"
    temp_file_path = os.path.join(temp_files_dir, temp_filename)

    # Create unique database path for this file
    db_path = f"./db_{unique_id}"
    collection_name = f"policy_{unique_id}"

    try:
        # Save uploaded file to temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"File saved to: {temp_file_path}")
        print("Processing with default questions...")

        start_time = time.time()

        # Initialize embedding model if not already done
        model = initialize_embedding_model()

        # Extract structured text from Excel
        print("Extracting structured text from Excel...")
        structured_blocks = extract_structured_text_from_excel(temp_file_path)

        if not structured_blocks:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract any data from the Excel file"
            )

        print(f"Extracted {len(structured_blocks)} blocks from Excel")

        # Create structural chunks
        print("Creating structural chunks...")
        structural_chunks = create_structural_chunks(structured_blocks)
        print(f"Created {len(structural_chunks)} structural chunks")

        # Add overlap to chunks
        print("Adding overlap to chunks...")
        overlapping_chunks = add_overlap_to_chunks(structural_chunks, overlap=CHUNK_OVERLAP)
        print(f"Created {len(overlapping_chunks)} overlapping chunks")

        # Store in vector database
        print("Storing chunks in vector database...")
        embedding_model = store_chunks_in_vector_db(
            overlapping_chunks,
            db_path=db_path,
            collection_name=collection_name,
            model_name=EMBEDDING_MODEL_NAME
        )

        # Define the default questions
        addon_question = "give me all the add-on & clauses from fire section?"
        occupancy_question = "give me occupancy?"
        
        # Define sum insured questions dynamically
        sum_insured_questions = [
            "from risk inspection details give me client location , and sum insured breakdown with their values in key value pair?"
        ]

        # Process addon_covers question
        print("Processing addon_covers question...")
        addon_chunks = retrieve_relevant_chunks(
            addon_question,
            embedding_model,
            db_path=db_path,
            collection_name=collection_name,
            top_k=15,
            rerank=True,
            verbose=False
        )

        addon_context = assemble_context(addon_question, addon_chunks, token_budget=25000)
        addon_answer = generate_response_with_bedrock(addon_question, addon_context,bedrock_client)

        # Process occupancy question
        print("Processing occupancy question...")
        occupancy_chunks = retrieve_relevant_chunks(
            occupancy_question,
            embedding_model,
            db_path=db_path,
            collection_name=collection_name,
            top_k=15,
            rerank=True,
            verbose=False
        )

        occupancy_context = assemble_context(occupancy_question, occupancy_chunks, token_budget=25000)
        occupancy_answer = generate_response_with_bedrock(occupancy_question, occupancy_context,bedrock_client)

        # Process sum insured questions dynamically
        print("Processing sum insured questions...")
        all_sum_insured_data = []
        
        for i, sum_insured_question in enumerate(sum_insured_questions):
            print(f"Processing sum insured question {i+1}: {sum_insured_question}")
            
            sum_insured_chunks = retrieve_relevant_chunks(
                sum_insured_question,
                embedding_model,
                db_path=db_path,
                collection_name=collection_name,
                top_k=15,
                rerank=True,
                verbose=False
            )

            sum_insured_context = assemble_context(sum_insured_question, sum_insured_chunks, token_budget=25000)
            sum_insured_answer = generate_response_with_bedrock(sum_insured_question, sum_insured_context,bedrock_client)

            # Parse the answer to structured JSON
            sum_insured_json = parse_sum_insured_to_structured_json(sum_insured_answer)
            
            # Add to the combined data
            all_sum_insured_data.extend(sum_insured_json)

        # Perform semantic matching for addon_covers only
        addon_matched_covers = {"matches": [], "semantic_matches": [], "unmatched": []}

        if os.path.exists(COVERS_JSON_PATH):
            try:
                # Initialize semantic matcher
                matcher = initialize_semantic_matcher()

                # Extract clauses from addon answer
                addon_clauses = extract_clauses_from_answer(addon_answer)
                print(f"Extracted {len(addon_clauses)} clauses from addon answer")

                if addon_clauses:
                    # Perform semantic matching
                    addon_matched_covers = matcher.compare_clauses_with_answers(COVERS_JSON_PATH, addon_clauses)

            except Exception as e:
                print(f"Error during semantic matching: {e}")
                addon_matched_covers = {
                    "matches": [],
                    "semantic_matches": [],
                    "unmatched": []
                }
        else:
            print(f"Warning: covers.json not found at {COVERS_JSON_PATH}")

        # Merge sections with same names from different questions
        merged_sections = {}
        for section_data in all_sum_insured_data:
            section_name = section_data["section"]
            if section_name not in merged_sections:
                merged_sections[section_name] = {"section": section_name, "locations": []}
            
            # Add locations from this section
            for location in section_data["locations"]:
                # Check if location already exists in this section
                existing_location = None
                for existing_loc in merged_sections[section_name]["locations"]:
                    if existing_loc["location"] == location["location"]:
                        existing_location = existing_loc
                        break
                
                if existing_location:
                    # Merge items from this location
                    for item in location["items"]:
                        # Check if item already exists
                        existing_item = None
                        for existing_item_obj in existing_location["items"]:
                            if existing_item_obj["description"] == item["description"]:
                                existing_item = existing_item_obj
                                break
                        
                        if existing_item:
                            # Sum the amounts if same item exists
                            existing_item["amount"] += item["amount"]
                        else:
                            # Add new item
                            existing_location["items"].append(item)
                else:
                    # Add new location
                    merged_sections[section_name]["locations"].append(location)
        
        # Convert merged sections back to array format
        final_sum_insured_data = list(merged_sections.values())

        processing_time = time.time() - start_time

        # Prepare final response in the requested format
        response_data = {
            "status": "success",
            "responses": {
                "addon_covers": {
                    "question": addon_question,
                    "answer": addon_answer,
                    "matched_covers": {
                        "matches": addon_matched_covers.get("matches", []),
                        "semantic_matches": addon_matched_covers.get("semantic_matches", [])
                    },
                    "unmatched_covers": addon_matched_covers.get("unmatched", [])
                },
                "Occupancy": {
                    "question": occupancy_question,
                    "answer": occupancy_answer
                },
                "sumInsuredSplit": {
                    "questions": sum_insured_questions,
                    "answer": final_sum_insured_data
                }
            }
        }

        print(f"File processed successfully in {processing_time:.2f} seconds")

        return JSONResponse(content=response_data)

    except HTTPException:
        # Re-raise HTTPExceptions as they are
        raise
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing the Excel file: {str(e)}"
        )

    finally:
        # Clean up temporary files and database
        cleanup_files = []
        cleanup_dirs = []

        # Add files and directories to cleanup lists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            cleanup_files.append(temp_file_path)

        if 'db_path' in locals() and os.path.exists(db_path):
            cleanup_dirs.append(db_path)

        # Clean up files
        for file_path in cleanup_files:
            try:
                os.remove(file_path)
                print(f"Cleaned up temporary file: {file_path}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up file {file_path}: {cleanup_error}")

        # Clean up directories
        for dir_path in cleanup_dirs:
            try:
                shutil.rmtree(dir_path)
                print(f"Cleaned up temporary database: {dir_path}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up database {dir_path}: {cleanup_error}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Excel Policy Chat API with Semantic Matching",
        "description": "Upload Excel files and ask questions about insurance policies with semantic clause matching",
        "endpoints": {
            "/upload-excel": "POST - Upload Excel file and ask a question",
            "/docs": "GET - API documentation"
        },
        "supported_formats": [".xlsx", ".xls"],
        "default_questions": [
            "give me occupancy?",
            "give me all the add-on & clauses from fire section?"
        ],
        "features": [
            "Automatic answers to default questions",
            "Semantic matching with covers.json for addon_covers",
            "Exact, fuzzy, and semantic clause matching",
            "Custom response format"
        ],
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedding_model_loaded": embedding_model is not None,
        "semantic_matcher_loaded": semantic_matcher is not None,
        "covers_json_exists": os.path.exists(COVERS_JSON_PATH),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server with semantic matching...")
    print(f"API will be available at: http://localhost:8001")
    print(f"Interactive docs at: http://localhost:8001/docs")
    print(f"Make sure covers.json exists at: {COVERS_JSON_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8001)