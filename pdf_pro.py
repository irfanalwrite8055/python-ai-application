import os
import re
import fitz  # PyMuPDF
import numpy as np
import uuid
from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb import PersistentClient
import camelot
import tabula
import time
import boto3
import json
from botocore.exceptions import ClientError
import random


# --- Configuration ---
EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5' # Good general-purpose embedding model
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2' # Effective reranking model
CHUNK_SIZE_TARGET = 500 # Target number of characters per chunk (approximate)
CHUNK_OVERLAP = 100    # Number of characters to overlap between chunks

# AWS Bedrock Configuration
AWS_REGION = 'ap-south-1'  # Change to your preferred region
BEDROCK_MODEL_ID = 'anthropic.claude-3-haiku-20240307-v1:0'  


def initialize_bedrock_client():
    """Initialize AWS Bedrock Runtime client"""
    try:
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION
        )
        print(f"AWS Bedrock client initialized successfully for region: {AWS_REGION}")
        return bedrock_runtime
    except Exception as e:
        print(f"Error initializing Bedrock client: {e}")
        print("Please ensure your AWS credentials are configured correctly.")
        return None

# Part 1: Text Extraction with Structure Preservation
def extract_structured_text(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text and structural information from PDF using PyMuPDF.
    Identifies blocks like headings, paragraphs, and tables
    based on heuristics. Includes page number and basic formatting.
    """
    doc = fitz.open(pdf_path)
    structured_blocks = []
    
    # -- PHASE 1: Extract tables using specialized libraries --
    extracted_tables = []
    table_bboxes_by_page = {}  # To track table locations and avoid duplicating content
    
    try:
        # Try to extract tables using Camelot (works well for bordered tables)
        tables_lattice = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        tables_stream = camelot.read_pdf(pdf_path, pages='all', flavor='stream', 
                                       edge_tol=50, line_scale=40)
        
        # Process lattice tables (tables with borders)
        for i, table in enumerate(tables_lattice):
            if table.df.empty:
                continue
                
            table_name = f"Table on page {table.page}"
            table_text = table.df.to_string(index=False)
            
            # Store table position to avoid duplicating content later
            if table.page not in table_bboxes_by_page:
                table_bboxes_by_page[table.page] = []
            
            # Use table's bbox if available, otherwise approximate from page info
            table_bbox = getattr(table, 'bbox', None)
            if not table_bbox:
                # Approximate bbox based on page dimensions
                page = doc[table.page-1]  # Camelot uses 1-indexed pages
                table_bbox = [0, 0, page.rect.width, page.rect.height]
            
            table_bboxes_by_page[table.page].append(table_bbox)
            
            extracted_tables.append({
                "text": table_text,
                "type": "table",
                "page": table.page,
                "font_size": 0,  # Not applicable for tables
                "bold": False,   # Not applicable for tables
                "bbox": table_bbox,
                "table_name": table_name
            })
            
        # Process stream tables (tables without clear borders)
        for i, table in enumerate(tables_stream):
            if table.df.empty:
                continue
                
            table_name = f"Table on page {table.page}"
            table_text = table.df.to_string(index=False)
            
            # Check if this table overlaps with one we already extracted
            is_duplicate = False
            if table.page in table_bboxes_by_page:
                # Simple text-based deduplication - could be improved
                for existing_table in extracted_tables:
                    if existing_table["page"] == table.page:
                        # Check for significant text overlap
                        if len(set(table_text.split()) & set(existing_table["text"].split())) / len(set(table_text.split())) > 0.7:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                if table.page not in table_bboxes_by_page:
                    table_bboxes_by_page[table.page] = []
                
                # Use table's bbox if available, otherwise approximate from page info
                table_bbox = getattr(table, 'bbox', None)
                if not table_bbox:
                    # Approximate bbox based on page dimensions
                    page = doc[table.page-1]  # Camelot uses 1-indexed pages
                    table_bbox = [0, 0, page.rect.width, page.rect.height]
                
                table_bboxes_by_page[table.page].append(table_bbox)
                
                extracted_tables.append({
                    "text": table_text,
                    "type": "table",
                    "page": table.page,
                    "font_size": 0,  # Not applicable for tables
                    "bold": False,   # Not applicable for tables
                    "bbox": table_bbox,
                    "table_name": table_name
                })
        
        # Try with tabula for tables that camelot might miss
        try:
            tabula_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            if tabula_tables:
                # Get document metadata to associate tables with page numbers
                doc_info = doc.metadata
                total_pages = len(doc)
                
                # Distribute tables across pages if tabula doesn't provide page numbers
                tables_per_page = max(1, len(tabula_tables) // total_pages)
                
                for i, df in enumerate(tabula_tables):
                    if df.empty:
                        continue
                        
                    # Estimate page number (tabula doesn't always provide it)
                    # This is a rough estimation - in real scenarios, you might need a better approach
                    estimated_page = min(total_pages, 1 + i // tables_per_page)
                    
                    table_text = df.to_string(index=False)
                    table_name = f"Table on page {estimated_page}"
                    
                    # Check if this table overlaps with one we already extracted
                    is_duplicate = False
                    for existing_table in extracted_tables:
                        # Check for significant text overlap
                        if len(set(table_text.split()) & set(existing_table["text"].split())) / len(set(table_text.split()) or [1]) > 0.7:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        # Approximate bbox based on page dimensions
                        page = doc[estimated_page-1]  # 0-indexed pages in PyMuPDF
                        table_bbox = [0, 0, page.rect.width, page.rect.height]
                        
                        if estimated_page not in table_bboxes_by_page:
                            table_bboxes_by_page[estimated_page] = []
                        table_bboxes_by_page[estimated_page].append(table_bbox)
                        
                        extracted_tables.append({
                            "text": table_text,
                            "type": "table",
                            "page": estimated_page,
                            "font_size": 0,  # Not applicable for tables
                            "bold": False,   # Not applicable for tables
                            "bbox": table_bbox,
                            "table_name": table_name
                        })
        except Exception as e:
            print(f"Tabula extraction error: {str(e)}")
            
    except Exception as e:
        print(f"Table extraction error: {str(e)}")
        # Continue with basic extraction if advanced libraries fail
    
    # -- PHASE 2: Extract text blocks with PyMuPDF (your original approach) --
    last_heading_by_page = {}  # Track the last heading seen on each page

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract text blocks with detailed information
        blocks = page.get_text("dict")["blocks"]
        
        # First pass to identify headings for potential table names
        for block in blocks:
            if block['type'] != 0:  # type 0 is text, other types are images/figures
                continue
                
            block_text = ""
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    if 4 < span['size'] < 30:  # Example range, adjust as needed
                        line_text += span["text"]
                if line_text.strip():
                    block_text += line_text.strip() + "\n"
            
            block_text = block_text.strip()
            if not block_text:
                continue
                
            # Detect if this is a heading
            is_bold = False
            font_size = 0
            if block["lines"] and block["lines"][0]["spans"]:
                first_span = block["lines"][0]["spans"][0]
                font_size = first_span["size"]
                if "bold" in first_span["font"].lower() or (first_span["flags"] & 1):
                    is_bold = True
            
            is_heading = False
            if (font_size > 11 and is_bold and len(block_text) < 150) or (font_size > 13 and len(block_text) < 200):
                if not block_text.endswith(('.', '!', '?')):
                    is_heading = True
            
            if re.match(r"^\s*(\d+(\.\d+)*|\([a-zA-Z]\)|\d+[.)])\s+", block_text.strip()):
                if len(block_text) < 200 and (is_bold or font_size > 11):
                    is_heading = True
                    
            if block_text.isupper() and len(block_text) > 3 and len(block_text) < 100 and not any(char.isdigit() for char in block_text):
                is_heading = True
                
            if is_heading:
                last_heading_by_page[page_num+1] = block_text
        
        # Second pass to extract content - enhanced for tables
        for block_idx, block in enumerate(blocks):
            if block['type'] != 0:  # Skip non-text blocks
                continue
                
            block_text = ""
            lines_text = []
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    if 4 < span['size'] < 30:  # Example range, adjust as needed
                        line_text += span["text"]
                line_text = line_text.strip()
                if line_text:
                    lines_text.append(line_text)
            
            block_text = "\n".join(lines_text).strip()
            if not block_text:
                continue
                
            # Check if this block is part of an already extracted table
            is_table_content = False
            current_page = page_num + 1  # Convert to 1-indexed for comparison with extracted tables
            
            if current_page in table_bboxes_by_page:
                block_bbox = block['bbox']
                for table_bbox in table_bboxes_by_page[current_page]:
                    # Check for significant overlap
                    # This is a simplified overlap check - could be improved
                    if (block_bbox[0] < table_bbox[2] and block_bbox[2] > table_bbox[0] and
                        block_bbox[1] < table_bbox[3] and block_bbox[3] > table_bbox[1]):
                        is_table_content = True
                        break
            
            if is_table_content:
                continue  # Skip this block as it's part of an extracted table
                
            # --- Table Detection Heuristics ---
            # Check if this block might be a table that wasn't caught by specialized libraries
            is_likely_table = False
            
            # Table indicators:
            # 1. Has consistent delimiters (|, tab, multiple spaces)
            has_delimiters = bool(re.search(r"(\||\t|    )", block_text))
            
            # 2. Has roughly consistent line lengths
            lines = block_text.split('\n')
            if len(lines) > 2:
                line_lengths = [len(line) for line in lines]
                length_variance = np.std(line_lengths) / np.mean(line_lengths) if np.mean(line_lengths) > 0 else 0
                consistent_lengths = length_variance < 0.3  # Low variance suggests table-like structure
            else:
                consistent_lengths = False
                
            # 3. Has aligned numeric columns (common in tables)
            numeric_patterns = [re.findall(r'\b\d+(?:\.\d+)?\b', line) for line in lines]
            has_numeric_columns = len(lines) > 2 and all(len(patterns) > 0 for patterns in numeric_patterns)
            
            is_likely_table = (has_delimiters and consistent_lengths) or (has_delimiters and has_numeric_columns)
            
            if is_likely_table:
                # Find an appropriate table name
                table_name = f"Table on page {page_num+1}"
                
                # Try to get table name from previous heading or caption
                if page_num+1 in last_heading_by_page:
                    heading_text = last_heading_by_page[page_num+1]
                    # Look for table indicators in the heading
                    if re.search(r"table|figure|tab\.|fig\.", heading_text, re.IGNORECASE):
                        table_name = heading_text
                
                # Add as a table block
                structured_blocks.append({
                    "text": block_text,
                    "type": "table",
                    "page": page_num + 1,
                    "font_size": 0,  # Not applicable for tables
                    "bold": False,   # Not applicable for tables
                    "bbox": block['bbox'],
                    "table_name": table_name
                })
                continue
            
            # --- Structure Heuristics (Original code) ---
            block_type = "paragraph"  # Default type
            is_bold = False
            font_size = 0
            
            # Get formatting from the first span of the first line
            if block["lines"] and block["lines"][0]["spans"]:
                first_span = block["lines"][0]["spans"][0]
                font_size = first_span["size"]
                if "bold" in first_span["font"].lower() or (first_span["flags"] & 1):
                    is_bold = True
            
            # Basic Heading Detection Heuristics
            is_heading = False
            if (font_size > 11 and is_bold and len(block_text) < 150) or (font_size > 13 and len(block_text) < 200):
                if not block_text.endswith(('.', '!', '?')):
                    is_heading = True
            
            if re.match(r"^\s*(\d+(\.\d+)*|\([a-zA-Z]\)|\d+[.)])\s+", block_text.strip()):
                if len(block_text) < 200 and (is_bold or font_size > 11):
                    is_heading = True
                    
            if block_text.isupper() and len(block_text) > 3 and len(block_text) < 100 and not any(char.isdigit() for char in block_text):
                is_heading = True
            
            # Determine heading level
            heading_level = 99
            if is_heading:
                if font_size > 14: heading_level = 1
                elif font_size > 12: heading_level = 2
                elif font_size > 11: heading_level = 3
                else: heading_level = 4
            
            if is_heading:
                block_type = f"heading_level_{heading_level}"
            
            # --- Add Block to Structured Blocks ---
            structured_blocks.append({
                "text": block_text,
                "type": block_type,
                "page": page_num + 1,
                "font_size": font_size,
                "bold": is_bold,
                "bbox": block['bbox']
            })
    
    # Add extracted tables to the structured blocks
    structured_blocks.extend(extracted_tables)
    
    # Sort all blocks by page number and vertical position
    structured_blocks.sort(key=lambda x: (x["page"], x.get("bbox", [0, 0, 0, 0])[1]))
    
    return structured_blocks

# Part 2: Structural Chunking
def create_structural_chunks(structured_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create chunks based on document structure, grouping related blocks.
    Assigns section/subsection metadata based on headings encountered.
    """
    chunks = []
    current_section = {"heading": "Document Start", "level": 0}
    current_subsection = {"heading": "", "level": 0}
    current_sub_subsection = {"heading": "", "level": 0} # Track potentially deeper levels

    # Buffer to accumulate paragraph content within a section/subsection
    paragraph_buffer = []
    buffer_page = None

    def flush_paragraph_buffer():
        """Helper to create a chunk from accumulated paragraph content."""
        nonlocal paragraph_buffer # Need to modify the outer scope variable
        if not paragraph_buffer:
            return

        combined_text = "\n".join([b["text"] for b in paragraph_buffer])

        # Determine the most relevant page number for the buffer
        # Could be the first page, last page, or a range. Let's use the first page for simplicity.
        buffer_first_page = paragraph_buffer[0]["page"] if paragraph_buffer else buffer_page

        chunks.append({
            "text": combined_text,
            "metadata": {
                "section": current_section["heading"],
                "subsection": current_subsection["heading"],
                "sub_subsection": current_sub_subsection["heading"],
                "type": "paragraph_content",
                "page": buffer_first_page # Associate chunk with its starting page
            }
        })
        paragraph_buffer = []

    for i, block in enumerate(structured_blocks):
        block_type = block["type"]
        block_text = block["text"]
        block_page = block["page"]

        # Update page for buffer if needed
        if buffer_page is None:
             buffer_page = block_page

        # Handle headings - flush buffer and update section tracking
        if block_type.startswith("heading_level_"):
            flush_paragraph_buffer() # Flush content before the new heading

            level = int(block_type.split("_")[-1])

            # Update section/subsection based on heading level
            if level == 1:
                current_section = {"heading": block_text, "level": level}
                current_subsection = {"heading": "", "level": 0}
                current_sub_subsection = {"heading": "", "level": 0}
            elif level == 2:
                current_subsection = {"heading": block_text, "level": level}
                current_sub_subsection = {"heading": "", "level": 0}
            elif level == 3:
                current_sub_subsection = {"heading": block_text, "level": level}
            # For levels > 3, they are sub_sub_sub sections, just update the current_sub_subsection name
            elif level > 3:
                 # Only update if the previous level tracking was higher or equal, otherwise might mess up hierarchy
                 if current_sub_subsection["level"] == 0 or level > current_sub_subsection["level"]:
                    current_sub_subsection = {"heading": block_text, "level": level}


            # Optional: Add heading as a separate, small chunk for direct retrieval
            # This helps if the user queries for the heading itself
            chunks.append({
                 "text": block_text,
                 "metadata": {
                     "section": current_section["heading"],
                     "subsection": current_subsection["heading"] if level > 1 else "",
                     "sub_subsection": current_sub_subsection["heading"] if level > 2 else "",
                     "type": block_type, # Keep the specific heading type
                     "page": block_page
                 }
            })
            buffer_page = None # Reset buffer page after a heading

        # Handle paragraph content
        elif block_type == "paragraph":
            paragraph_buffer.append(block)
            # Simple buffer flushing logic: flush if buffer is large or we encounter a heading/end of document soon
            # A more sophisticated approach could check for sentence boundaries or topic shifts
            # For now, flush if the buffer gets too long or the next block is a heading
            buffer_text_length = sum(len(b["text"]) for b in paragraph_buffer)
            next_block_is_heading = (i + 1 < len(structured_blocks) and
                                     structured_blocks[i+1]["type"].startswith("heading_level_"))

            if buffer_text_length > CHUNK_SIZE_TARGET or next_block_is_heading:
                 flush_paragraph_buffer()
                 buffer_page = None # Reset buffer page after flushing

        # Add other potential block types here if needed (e.g., lists, table text if not using a table parser)
        # For now, they are treated as paragraphs by the default `block_type`.


    # Flush any remaining content at the end of the document
    flush_paragraph_buffer()

    return chunks

# Part 3: Add Overlap to Chunks
def add_overlap_to_chunks(chunks: List[Dict[str, Any]], overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Adds overlapping text from the end of a chunk to the beginning of the next.
    Considers section boundaries to avoid overlapping unrelated content.
    """
    overlapping_chunks = []

    for i, chunk in enumerate(chunks):
        overlapping_chunks.append(chunk) # Add the original chunk

        # Add overlap to the next chunk if it exists and is in the same section/subsection
        if i + 1 < len(chunks):
            next_chunk = chunks[i+1]

            # Check if chunks are in the same logical section/subsection
            same_section = (chunk["metadata"].get("section") == next_chunk["metadata"].get("section") and
                            chunk["metadata"].get("subsection") == next_chunk["metadata"].get("subsection"))
            # Also avoid overlapping across heading chunks themselves
            if not same_section or next_chunk["metadata"].get("type").startswith("heading_level_"):
                continue # Do not overlap across sections or into headings

            # Get overlap text from the end of the current chunk
            overlap_text = chunk["text"][-overlap:].strip()

            if overlap_text:
                 # Prepend overlap text to the beginning of the next chunk's text
                 # Ensure there's a clear separator
                 next_chunk["text"] = overlap_text + "\n---\n" + next_chunk["text"]

    # Note: This modifies the text of the *original* chunks list for the next iteration.
    # If you need the original non-overlapping chunks later, work on a copy.
    # For this pipeline, modifying in place is fine as the 'chunks' list
    # is processed sequentially and then discarded after this step.

    return overlapping_chunks # This list now contains the original chunks with modified text for overlap


# Part 4: Vector Database Storage (Uses the refined chunks)
def store_chunks_in_vector_db(chunks: List[Dict[str, Any]], db_path: str = "./chroma_db",
                             collection_name: str = "insurance_policy",
                             model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Store chunks in a vector database with their embeddings.
    Uses the SentenceTransformer model specified.
    """
    print(f"Initializing embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Initializing ChromaDB client at: {db_path}")
    chroma_client = PersistentClient(path=db_path)

    print(f"Getting or creating collection: {collection_name}")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Clear existing data if reindexing (handled by `force_reindex` in main)
    # The get_or_create handles the collection existence; we assume if we reach here
    # after force_reindex=True, the old collection was handled (e.g., deleted in main)
    # If not force_reindex and collection exists, we append. This might lead to duplicates
    # if the same PDF is processed again without clearing. A robust system would track document IDs.
    # For simplicity here, we'll rely on force_reindex to clear.

    # Prepare data for storage
    texts = []
    metadatas = []
    ids = []

    for chunk in chunks:
        texts.append(chunk["text"])
        metadatas.append(chunk["metadata"])
        # Generate a unique ID for each chunk
        ids.append(str(uuid.uuid4()))

    if not texts:
        print("No chunks to store.")
        return model

    print(f"Generating embeddings for {len(texts)} chunks and adding to database...")
    # Generate embeddings in batches to optimize GPU usage
    batch_size = 32 # Adjust based on your GPU memory
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        batch_texts = texts[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]

        # Generate embeddings
        try:
            batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
             # Convert tensor to numpy, then to list for ChromaDB
            batch_embeddings_list = batch_embeddings.cpu().numpy().tolist()
        except Exception as e:
            print(f"Error generating embeddings for batch {i//batch_size}: {e}")
            print("Skipping this batch.")
            continue # Skip adding this batch if embedding fails

        # Add to database
        try:
            collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=batch_embeddings_list
            )
        except Exception as e:
            print(f"Error adding batch {i//batch_size} to ChromaDB: {e}")
            print("Skipping this batch.")
            continue # Skip adding this batch if ChromaDB add fails

    print("Finished storing chunks.")
    return model

# Part 5: Query Processing and Retrieval (Uses the defined models)
def retrieve_relevant_chunks(query: str,
                              model: SentenceTransformer,
                              db_path: str = "./chroma_db",
                              collection_name: str = "insurance_policy",
                              top_k: int = 15,
                              rerank: bool = True,
                              verbose: bool = True) -> list:
    """
    Retrieve top-k relevant chunks using embeddings and reranking.
    """
    # Connect to ChromaDB
    chroma_client = PersistentClient(path=db_path)
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        print(f"Error: Collection '{collection_name}' not found in '{db_path}'. Please run indexing first.")
        return [] # Return empty list if collection doesn't exist

    # Create query embedding
    query_embedding = model.encode(query).tolist()

    # Retrieve initial candidates
    # Retrieve more candidates than top_k if reranking is enabled
    initial_k = min(50, top_k * 3) if rerank else top_k
    # Ensure we don't request more results than are in the collection
    count = collection.count()
    if count == 0:
        print("Database is empty.")
        return []
    initial_k = min(initial_k, count)


    if verbose:
        print(f"Retrieving initial {initial_k} candidates...")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k,
        include=["documents", "metadatas", "distances"] # Include distances for potential analysis
    )

    retrieved_chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            retrieved_chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] # Include distance
            })

    if verbose:
        print(f"Retrieved {len(retrieved_chunks)} initial chunks.")
        # print("\n[Debug] Top initial retrieved chunks (first 300 chars):")
        # for i, chunk in enumerate(retrieved_chunks[:min(5, len(retrieved_chunks))]):
        #     print(f"\n--- Chunk {i+1}, Distance: {chunk['distance']:.4f} ---")
        #     print(f"Section: {chunk['metadata'].get('section', 'N/A')} > {chunk['metadata'].get('subsection', 'N/A')}")
        #     print(f"Page: {chunk['metadata'].get('page', 'N/A')}")
        #     print(chunk['text'][:300] + "...\n")

    # Cross-Encoder Reranking
    if rerank and len(retrieved_chunks) > top_k:
        if verbose:
            print(f"Reranking {len(retrieved_chunks)} chunks...")
        try:
            reranker = CrossEncoder(RERANKER_MODEL_NAME)
            pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]
            scores = reranker.predict(pairs)

            scored_chunks = list(zip(retrieved_chunks, scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True) # Sort by reranker score

            retrieved_chunks = [chunk for chunk, _ in scored_chunks[:top_k]]

            if verbose:
                 print(f"Finished reranking, keeping top {top_k}.")
                 # print("\n[Debug] Top reranked chunks (first 300 chars):")
                 # for i, chunk in enumerate(retrieved_chunks):
                 #    print(f"\n--- Reranked Chunk {i+1} ---")
                 #    print(f"Section: {chunk['metadata'].get('section', 'N/A')} > {chunk['metadata'].get('subsection', 'N/A')}")
                 #    print(f"Page: {chunk['metadata'].get('page', 'N/A')}")
                 #    print(chunk['text'][:300] + "...\n")

        except Exception as e:
            print(f"Error during reranking: {e}. Skipping reranking.")
            # Fallback to returning the top_k from initial retrieval based on distance
            retrieved_chunks.sort(key=lambda x: x["distance"])
            retrieved_chunks = retrieved_chunks[:top_k]


    return retrieved_chunks


# Part 6: Knowledge Filtering and Context Assembly
def assemble_context(query: str,
                     chunks: List[Dict[str, Any]],
                     token_budget: int = 10000) -> str: # Set a reasonable token budget for context
    """
    Assemble a context from retrieved chunks, prioritizing informative chunks
    while staying within the token budget. Includes metadata in the context.
    """
    if not chunks:
        return "No relevant information found in the policy document."

    # Sort chunks by page number and then by structural type (headings first)
    # This helps present the context in a more logical document order
    def sort_key(chunk):
        # Assign a numerical value based on chunk type for sorting (headings first)
        type_order = {
            "heading_level_1": 1,
            "heading_level_2": 2,
            "heading_level_3": 3,
            "heading_level_4": 4,
            "paragraph_content": 5,
            "table_content": 6, # If table content was explicitly typed
        }.get(chunk["metadata"].get("type", "paragraph_content"), 99) # Default to paragraph_content order

        # Use page number, then type order for sorting
        return (chunk["metadata"].get("page", 0), type_order)

    chunks.sort(key=sort_key)


    # Estimate tokens (using a more common approximation: 1 word ≈ 1.3 tokens)
    def estimate_tokens(text):
        return len(text.split()) * 1.3

    context_parts = []
    current_tokens = 0

    # Add query to estimate its token size (needed for the full prompt later)
    # This isn't added to the context itself, but helps in budget calculation
    query_tokens = estimate_tokens(query)
    # Also account for prompt template overhead (rough estimate)
    prompt_overhead_tokens = 500 # Estimate for system prompt and template structure

    available_budget = token_budget - query_tokens - prompt_overhead_tokens
    if available_budget <= 0:
        print("Warning: Token budget is too small to include context with the query and prompt overhead.")
        available_budget = 100 # Allow at least a small buffer


    for chunk in chunks:
        # Format chunk with metadata (Page number and section/subsection)
        metadata_line = f"[Page {chunk['metadata'].get('page', 'N/A')}]"
        section_info = []
        if chunk["metadata"].get("section"):
            section_info.append(chunk["metadata"]["section"])
        if chunk["metadata"].get("subsection"):
            section_info.append(chunk["metadata"]["subsection"])
        if chunk["metadata"].get("sub_subsection"):
             section_info.append(chunk["metadata"]["sub_subsection"])

        if section_info:
             metadata_line += " [" + " > ".join(section_info) + "]"


        formatted_chunk = f"{metadata_line}\n{chunk['text']}"
        chunk_tokens = estimate_tokens(formatted_chunk)

        # Check if adding this chunk exceeds the budget
        if current_tokens + chunk_tokens > available_budget:
            # If we already have some context, stop adding more
            if context_parts:
                break
            # If this is the first chunk and it's too large, add a truncated version
            else:
                # Truncate the first chunk to fit within the budget
                # A more sophisticated approach would truncate preserving sentence boundaries
                max_chars = int((available_budget - current_tokens) * 3.5) # Approx tokens to chars
                truncated_text = chunk["text"][:max_chars]
                formatted_chunk = f"{metadata_line}\n{truncated_text}...\n[Context truncated due to length]"
                context_parts.append(formatted_chunk)
                current_tokens += estimate_tokens(formatted_chunk)
                break # Stop after adding the truncated first chunk

        else:
            current_tokens += chunk_tokens
            context_parts.append(formatted_chunk)

    if not context_parts:
        return "Relevant information was retrieved, but could not be included in the context due to token budget constraints."


    return "\n\n".join(context_parts)

# Part 7: LLM Response Generation
# Reuse the existing generate_response function, just update the n_ctx
def generate_response_with_bedrock(query: str, context: str, bedrock_client, max_retries: int = 5) -> str:
    """Generate a response using AWS Bedrock with enhanced retry logic and rate limiting"""
    if not bedrock_client:
        return "Error: Bedrock client not initialized. Please check your AWS configuration."

    system_prompt = """CRITICAL: Start every response immediately with the requested data. Never use introductory phrases.

You are an expert insurance underwriter with extensive knowledge of insurance terminology,
   policy structures, and legal interpretations. Your task is to provide accurate, helpful responses to questions
   about insurance policies by carefully analyzing the provided policy text.
   
   RESPONSE RULES - FOLLOW EXACTLY:
   - Start immediately with the answer
   - Never use "The occupancy description for this policy is:", "Based on", "Here is", "The answer is", or any introductory text
   - Just provide the raw data requested
   - No explanations, no context, no lead-ins
   
   GUIDELINES:
   1. Base your answers EXCLUSIVELY on the provided policy context. Do not introduce information from outside the context.
   2. If the information needed to answer the question is not explicitly stated in the context, clearly state that you cannot
      find this information in the provided policy text.
   3. For questions about definitions, provide the exact definition from the policy if available in the context.
   4. Present information in a structured, easy-to-understand format while maintaining accuracy based on the context.
   5. Provide ONLY the specific information requested - nothing more. Your responses should be direct and concise.
   6. For data points like sums insured, names, dates, or other factual information, extract and present ONLY that exact information.
   7. If answering a complex question requiring interpretation, explain your reasoning based *solely* on the policy language provided.
   8. Do not include explanations, context, additional details, or surrounding text unless specifically asked.
   
   FORMATTING REQUIREMENTS:
   - Do not use bullet points, asterisks, or special characters for lists
   - Present monetary values as plain numbers without currency symbols or commas
   - Present information in simple format: category followed by colon and value
   - Use simple line breaks between items
   - Keep responses clean and minimal
   - Never use numbered lists (1., 2., 3.)
   - Extract ALL data points that exist in the source material, never omit any
   - Be completely consistent for identical question types
   
   OUTPUT ONLY THE REQUESTED DATA. NO INTRODUCTORY TEXT WHATSOEVER.
   """

    # Prepare the message for Claude
    messages = [
        {
            "role": "user",
            "content": f"""Context from Insurance Policy:
{context}

Question: {query}"""
        }
    ]

    # Prepare the request body for Claude
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
        "system": system_prompt,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.95
    }

    # Enhanced retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            print(f"Making Bedrock API call (attempt {attempt + 1}/{max_retries})...")
            
            # Make request to Bedrock
            response = bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps(body),
                contentType='application/json'
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            answer = response_body['content'][0]['text'].strip()

            # Post-process to remove excessive pipe characters from poorly formatted tables
            if answer.count("|") > 10:
                lines = answer.split("\n")
                processed_lines = []

                for line in lines:
                    cleaned_line = re.sub(r'\|\s*\|+', ' ', line)
                    cleaned_line = re.sub(r'^\s*\|\s*|\s*\|\s*$', '', cleaned_line)
                    if cleaned_line.strip():
                        processed_lines.append(cleaned_line)

                answer = "Based on the insurance policy information:\n\n" + "\n".join(processed_lines)

            print("✅ Bedrock API call successful!")
            return answer

        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"❌ Bedrock API error (attempt {attempt + 1}): {error_code}")
            
            if error_code == 'ThrottlingException':
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Rate limited. Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Error: Rate limit exceeded after {max_retries} attempts. Please wait a few minutes and try again."
            
            elif error_code in ['ValidationException', 'ModelNotReadyException']:
                if attempt < max_retries - 1:
                    wait_time = 2 + (attempt * 0.5)
                    print(f"⏳ Model/validation error. Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Error: {error_code} - {e.response['Error']['Message']}"
            
            else:
                # For other errors, don't retry
                return f"Error: {error_code} - {e.response['Error']['Message']}"
        
        except Exception as e:
            print(f"❌ Unexpected error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 1 + (attempt * 0.5)
                print(f"⏳ Unexpected error. Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
                continue
            else:
                return f"An unexpected error occurred after {max_retries} attempts: {str(e)}"

    return "Error: Maximum retry attempts exceeded."


# Part 8: Process Insurance PDF Query (Uses the updated functions)
def process_insurance_pdf_query(pdf_path: str,
                               query: str,
                               bedrock_client,  # Changed from model_path
                               db_path: str = "./chroma_db",
                               collection_name: str = "insurance_policy",
                               force_reindex: bool = False) -> Dict[str, Any]:
    """
    End-to-end process for answering queries about insurance policy PDFs using Bedrock.
    """
    chroma_client = PersistentClient(path=db_path)

    # Check if indexing is needed
    index_needed = force_reindex
    if not force_reindex:
        try:
            collection = chroma_client.get_collection(name=collection_name)
            if collection.count() == 0:
                print("Existing collection is empty, indexing is required.")
                index_needed = True
            else:
                 print(f"Using existing index with {collection.count()} chunks.")
                 index_needed = False
        except:
            print(f"Collection '{collection_name}' not found, indexing is required.")
            index_needed = True

    embedding_model = None

    # Extract and index if needed (same as before)
    if index_needed:
        print("Starting file indexing process...")
        if force_reindex:
            try:
                chroma_client.delete_collection(name=collection_name)
                print(f"Deleted existing collection '{collection_name}' for reindexing.")
            except:
                pass

        start_time = time.time()

        print("Extracting structured text from file...")
        structured_blocks = extract_structured_text(pdf_path)
        print(f"  Extracted {len(structured_blocks)} text blocks.")
        if not structured_blocks:
             print("No text blocks extracted. Cannot proceed with indexing.")
             return {"query": query, "answer": "Error: Could not extract text from the file.", "context_chunks": [], "context_used": ""}

        print("Applying structural chunking...")
        structural_chunks = create_structural_chunks(structured_blocks)
        print(f"  Created {len(structural_chunks)} initial structural chunks.")

        print(f"Adding {CHUNK_OVERLAP} character overlap to chunks...")
        overlapping_chunks = add_overlap_to_chunks(structural_chunks, overlap=CHUNK_OVERLAP)
        print(f"  Resulting chunks after overlap processing: {len(overlapping_chunks)}")

        print("Storing chunks in vector database...")
        embedding_model = store_chunks_in_vector_db(
            overlapping_chunks,
            db_path=db_path,
            collection_name=collection_name,
            model_name=EMBEDDING_MODEL_NAME
        )
        index_time = time.time() - start_time
        print(f"Indexing completed in {index_time:.2f} seconds.")
    else:
        try:
             print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
             embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
             print(f"Error loading embedding model: {e}. Cannot process queries.")
             return {"query": query, "answer": "Error: Could not load the embedding model.", "context_chunks": [], "context_used": ""}

    # Process query (same retrieval logic)
    print(f"\nProcessing query: \"{query}\"")

    relevant_chunks = retrieve_relevant_chunks(
        query,
        embedding_model,
        db_path=db_path,
        collection_name=collection_name,
        rerank=True
    )

    if not relevant_chunks:
        print("No relevant chunks found for the query.")
        context = assemble_context(query, relevant_chunks)
        answer = generate_response_with_bedrock(query, context, bedrock_client)  # Use Bedrock
        return {
            "query": query,
            "answer": answer,
            "context_chunks": relevant_chunks,
            "context_used": context
        }

    # Assemble context for LLM
    context_token_budget = 30000 - (len(query.split()) * 1.3) - 1000  # Claude context window
    context = assemble_context(query, relevant_chunks, token_budget=int(context_token_budget))

    # Generate response using Bedrock
    answer = generate_response_with_bedrock(query, context, bedrock_client)  # Use Bedrock

    return {
        "query": query,
        "answer": answer,
        "context_chunks": relevant_chunks,
        "context_used": context
    }

# Part 9: Main Function
def main():
    """
    Main function to handle the flow using AWS Bedrock with rate limiting
    """
    # File path (same as before)
    pdf_path = "/home/yash/Desktop/fast/Fire_Policy .PDF"
    
    # No need for local model path anymore
    db_path = "./insurance_policy_db"
    collection_name = "insurance_policy"
    force_reindex = False

    # Initialize Bedrock client
    bedrock_client = initialize_bedrock_client()
    if not bedrock_client:
        print("Failed to initialize Bedrock client. Exiting.")
        return

    # Validate file path
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    print(f"Insurance Policy Chat System with AWS Bedrock")
    print(f"================================================")
    print(f"File: {pdf_path}")
    print(f"Bedrock Model: {BEDROCK_MODEL_ID}")
    print(f"AWS Region: {AWS_REGION}")
    print(f"Vector Database: {db_path}")
    print(f"Collection: {collection_name}")
    print(f"Force Reindex: {force_reindex}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Reranker Model: {RERANKER_MODEL_NAME}")
    print(f"Chunk Target Size: {CHUNK_SIZE_TARGET} chars")
    print(f"Chunk Overlap: {CHUNK_OVERLAP} chars")
    print(f"================================================")

    # Process file indexing (if needed)
    print("Checking/Performing file indexing...")
    try:
        file_ext = os.path.splitext(pdf_path)[1].lower()
        
        if file_ext in ['.xlsx', '.xls']:
            print("Processing Excel file...")
            structured_blocks = extract_structured_text(pdf_path)
            print(f"  Extracted {len(structured_blocks)} blocks from Excel.")
            
            if not structured_blocks:
                print("No blocks extracted. Cannot proceed with indexing.")
                return
                
            print("Applying structural chunking...")
            structural_chunks = create_structural_chunks(structured_blocks)
            print(f"  Created {len(structural_chunks)} structural chunks.")
            
            print(f"Adding {CHUNK_OVERLAP} character overlap to chunks...")
            overlapping_chunks = add_overlap_to_chunks(structural_chunks, overlap=CHUNK_OVERLAP)
            print(f"  Resulting chunks after overlap processing: {len(overlapping_chunks)}")
            
            print("Storing chunks in vector database...")
            if force_reindex:
                try:
                    chroma_client = PersistentClient(path=db_path)
                    chroma_client.delete_collection(name=collection_name)
                    print(f"Deleted existing collection '{collection_name}' for reindexing.")
                except Exception as e:
                    print(f"Note: {e}")
                    
            embedding_model = store_chunks_in_vector_db(
                overlapping_chunks,
                db_path=db_path,
                collection_name=collection_name,
                model_name=EMBEDDING_MODEL_NAME
            )
            print("Indexing completed successfully.")
            
        print("Indexing check/completion finished.")
    except Exception as e:
        print(f"An error occurred during the initial indexing check/process: {e}")
        import traceback
        traceback.print_exc()
        print("Please check the file path and ensure dependencies are installed.")
        return

    # Interactive query loop with rate limiting
    print("\nInsurance Policy Chat System Ready!")
    print("Type 'exit' or 'quit' to end the session")
    print("⚠️  Note: There's a 2-second delay between questions to avoid rate limits")

    question_count = 0
    
    while True:
        query = input("\nEnter your question about the insurance policy: ")

        if query.lower() in ['exit', 'quit']:
            print("Exiting chat system. Goodbye!")
            break

        if not query.strip():
            continue

        # Add delay between questions (except for the first one)
        if question_count > 0:
            print("⏳ Waiting 2 seconds to avoid rate limits...")
            time.sleep(2)
        
        question_count += 1

        start_time = time.time()
        try:
            result = process_insurance_pdf_query(
                pdf_path=pdf_path,
                query=query,
                bedrock_client=bedrock_client,  # Pass Bedrock client instead of model path
                db_path=db_path,
                collection_name=collection_name,
                force_reindex=False
            )

            query_time = time.time() - start_time

            print("\n" + "="*50)
            print("ANSWER:")
            print("-"*50)
            print(result["answer"])
            print("="*50)
            print(f"Response generated in {query_time:.2f} seconds using AWS Bedrock")

            show_context = input("\nWould you like to see the context used? (y/n): ")
            if show_context.lower() == 'y':
                print("\nCONTEXT USED:")
                print("-"*50)
                print(result["context_used"])
                print("-"*50)

        except Exception as e:
            print(f"\nAn error occurred while processing your query: {str(e)}")
            import traceback
            traceback.print_exc()


# Alternative: Batch processing function for multiple questions
def process_multiple_questions_batch(pdf_path: str, questions: list, bedrock_client, 
                                   db_path: str = "./chroma_db", 
                                   collection_name: str = "insurance_policy"):
    """
    Process multiple questions with proper rate limiting between them
    """
    results = []
    
    for i, question in enumerate(questions):
        print(f"\n📝 Processing question {i+1}/{len(questions)}: {question}")
        
        # Add delay between questions (except for the first one)
        if i > 0:
            print("⏳ Waiting 3 seconds to avoid rate limits...")
            time.sleep(3)
        
        try:
            result = process_insurance_pdf_query(
                pdf_path=pdf_path,
                query=question,
                bedrock_client=bedrock_client,
                db_path=db_path,
                collection_name=collection_name,
                force_reindex=False
            )
            results.append(result)
            print(f"✅ Question {i+1} completed successfully")
            
        except Exception as e:
            print(f"❌ Error processing question {i+1}: {e}")
            results.append({
                "query": question,
                "answer": f"Error: {str(e)}",
                "context_chunks": [],
                "context_used": ""
            })
    
    return results

if __name__ == "__main__":
    main()