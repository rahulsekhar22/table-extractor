import os
import json
import re
import base64
import logging
from typing import Optional, Dict, Any, List

import fitz  # PyMuPDF
# The following imports might not be strictly necessary for the core logic
# but were in your original code, so I'll keep them if they are used elsewhere.
# from io import BytesIO
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

# Assuming these are available from your environment setup
from azure_client import get_azure_openai_client
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import jsonschema for strict validation
from jsonschema import validate, ValidationError

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- JSON Schema Definition ---
# This schema is critical for deterministic output and validation
TABLE_EXTRACTION_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ExtractedTables",
  "description": "Schema for extracted tables from financial documents.",
  "type": "object",
  "properties": {
    "tables": {
      "type": "array",
      "description": "An array of identified tables.",
      "items": {
        "type": "object",
        "properties": {
          "table_name": {
            "type": "string",
            "description": "The full numbered title of the table (e.g., 'Table 3: Income Statement')."
          },
          "metadata": {
            "type": "string",
            "description": "A concise, standardized description of the table's content type (e.g., 'Balance Sheet', 'Income Statement', 'Cash Flow', 'Operating Expenses', 'Sales Data', 'Summary Financials')."
          },
          "headers": {
            "type": "array",
            "description": "An ordered list of all column headers, left-to-right, exactly as seen.",
            "items": { "type": "string" }
          },
          "rows": {
            "type": "array",
            "description": "An array of data rows. Each inner array represents a row.",
            "items": {
              "type": "array",
              "description": "A single row of data, where each element is a cell value.",
              "items": { "type": "string" }
            }
          }
        },
        "required": ["table_name", "metadata", "headers", "rows"],
        "additionalProperties": False # Crucial: Disallow extra keys
      }
    }
  },
  "required": ["tables"],
  "additionalProperties": False # Crucial: Disallow extra top-level keys
}

class PDFTableExtractor:
    def __init__(self, max_retries: int = 3, temperature: float = 0.0):
        # Initialize Azure OpenAI client
        self.client: AzureChatOpenAI = get_azure_openai_client()
        if self.client is None:
            raise ValueError("Failed to initialize Azure OpenAI client")
        self.max_retries = max_retries
        self.temperature = temperature # Keep at 0 for determinism

    def convert_pdf_to_base64(self, pdf_path: str, dpi: int = 300) -> List[str]:
        """
        Convert all pages of a PDF to base64 encoded images using PyMuPDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            dpi (int): Dots per inch for rendering resolution. Higher DPI for better table recognition.
            
        Returns:
            list: List of base64 encoded images for each page
        """
        try:
            # Open the PDF file
            pdf_document = fitz.open(pdf_path)
            base64_images = []
            
            for page_num, page in enumerate(pdf_document):
                # Calculate zoom based on desired DPI (default PyMuPDF is 72 DPI)
                zoom = dpi / 72 
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert pixmap to bytes in PNG format
                img_data = pix.tobytes("png")
                
                # Convert directly to base64
                img_str = base64.b64encode(img_data).decode()
                base64_images.append(img_str)
            
            pdf_document.close()
            logging.info(f"Successfully converted {len(base64_images)} pages from {pdf_path} to base64 images at {dpi} DPI.")
            return base64_images
        except Exception as e:
            logging.error(f"Error converting PDF '{pdf_path}' to base64: {e}", exc_info=True)
            raise Exception(f"Error converting PDF to base64: {str(e)}")

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Robustly parses the LLM response to extract a JSON object.
        Handles cases where JSON is wrapped in markdown or embedded in text.
        """
        cleaned_text = response_text.strip()
        
        # Attempt to remove common markdown wrappers
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:].strip()
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3].strip()
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:].strip()
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3].strip()

        try:
            result = json.loads(cleaned_text)
            return result
        except json.JSONDecodeError:
            logging.warning("Initial JSON parse failed. Attempting substring-based extraction.")
            # Fallback: try to find the first '{' and last '}'
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    json_str = cleaned_text[start_idx : end_idx + 1]
                    result = json.loads(json_str)
                    logging.info("Successfully extracted and parsed JSON from malformed response via substring.")
                    return result
                except json.JSONDecodeError as e:
                    logging.error(f"Failed substring-based JSON extraction: {e}")
            raise ValueError("Could not parse or extract valid JSON from LLM response.")

    def _validate_json_output(self, data: Dict[str, Any], page_number: int) -> None:
        """
        Validates the extracted JSON against the schema and performs custom logical checks.
        Raises ValueError if validation fails.
        """
        try:
            validate(instance=data, schema=TABLE_EXTRACTION_SCHEMA)
            logging.debug(f"Page {page_number}: JSON output passed schema validation.")
        except ValidationError as e:
            logging.warning(f"Page {page_number}: JSON schema validation failed: {e.message} at {e.path}")
            raise ValueError(f"LLM output did not conform to schema: {e.message}")
        
        # Custom logical validations
        tables = data.get("tables", [])
        for i, table in enumerate(tables):
            table_name = table.get("table_name", f"Table {i+1}")
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            
            if not isinstance(headers, list) or not all(isinstance(h, str) for h in headers):
                raise ValueError(f"Page {page_number}, Table '{table_name}': Headers must be a list of strings.")
            if not isinstance(rows, list):
                raise ValueError(f"Page {page_number}, Table '{table_name}': Rows must be a list.")

            num_headers = len(headers)
            for j, row in enumerate(rows):
                if not isinstance(row, list) or not all(isinstance(c, str) for c in row):
                     raise ValueError(f"Page {page_number}, Table '{table_name}', Row {j+1}: Row data must be a list of strings.")
                if len(row) != num_headers:
                    logging.warning(
                        f"Page {page_number}, Table '{table_name}', Row {j+1}: "
                        f"Row length mismatch. Expected {num_headers} columns, but got {len(row)}. "
                        f"Row: {row}"
                    )
                    # For determinism, we can raise an error here to force a retry
                    raise ValueError(
                        f"Page {page_number}, Table '{table_name}': Row length mismatch. "
                        f"Expected {num_headers} columns, got {len(row)} in row {j+1}."
                    )
        logging.debug(f"Page {page_number}: JSON output passed custom logical validations.")


    def analyze_with_vision(self, image_base64: str, page_number: int = 1) -> Dict[str, Any]:
        """
        Analyze the image using Azure OpenAI's vision model to identify and structure tables.
        Includes retry logic for robustness.
        """
        system_prompt = f"""You are a table extraction assistant for financial documents. Extract ALL tables from images and return valid JSON only.

Output JSON Schema:
{json.dumps(TABLE_EXTRACTION_SCHEMA, indent=2)}

Instructions:
1. Return only valid JSON - no markdown, no extra text
2. Extract ALL tables that have numbered titles (e.g., "Table 1:", "Table 2: Description", "Table 3 - Summary")
3. Look for any pattern: "Table X", "Table X:", "Table X -", "Table X.", where X is a number
4. Extract complete cell values exactly as shown - use "" for empty cells
5. Each row must have same number of cells as headers
6. If no numbered tables exist, return {{"tables": []}}
7. Include complete table_name and brief metadata description

Important: Extract ALL numbered tables you find - do not skip any tables."""

        user_content = [
            {
                "type": "text",
                "text": """Extract ALL tables with numbered titles from this image. Do not skip any tables.

For each table found:
- Extract complete table title with number
- Extract all column headers left-to-right
- Extract all data rows with same number of cells as headers
- Use "" for empty cells
- Add brief metadata description

Scan the entire image carefully and extract every numbered table. Return only the JSON object."""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": "high"
                }
            }
        ]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]

        for attempt in range(self.max_retries):
            logging.info(f"Page {page_number}: Attempt {attempt + 1}/{self.max_retries} to analyze with vision model.")
            try:
                response = self.client.invoke(messages, temperature=self.temperature)
                response_text = response.content.strip()
                
                logging.debug(f"Page {page_number}, Attempt {attempt + 1}: Raw LLM Response: {response_text[:1000]}...") # Log first 1000 chars

                # Parse the response to get a JSON object
                parsed_result = self._parse_llm_response(response_text)
                
                # Validate the parsed JSON against the schema and custom rules
                self._validate_json_output(parsed_result, page_number)
                
                logging.info(f"✅ Page {page_number}: Successfully extracted and validated {len(parsed_result.get('tables', []))} table(s).")
                return parsed_result
                
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                logging.warning(f"Page {page_number}, Attempt {attempt + 1}: Validation/Parse Error: {e}")
                if attempt == self.max_retries - 1:
                    logging.error(f"❌ Page {page_number}: Max retries reached. Returning empty tables list for this page.")
                    return {"tables": []}
                # Optional: Add a small delay before retrying
                # import time
                # time.sleep(1) 
            except Exception as e:
                logging.error(f"Page {page_number}, Attempt {attempt + 1}: An unexpected error occurred during vision analysis: {e}", exc_info=True)
                if attempt == self.max_retries - 1:
                    logging.error(f"❌ Page {page_number}: Max retries reached. Returning empty tables list for this page.")
                    return {"tables": []}
                # time.sleep(1) # Optional delay

        return {"tables": []} # Should ideally not be reached, but as a fallback

    @staticmethod
    def clean_numeric_value(value: str) -> str:
        """
        Clean a numeric value by removing currency symbols, commas, and handling special cases.
        
        Rules:
        - Remove: $, commas (,)
        - If number is in parentheses (), it's negative (e.g., (100) -> -100)
        - If empty or contains only '-', replace with '0'
        - Preserve the first column (row headers) as-is
        
        Args:
            value (str): The value to clean
            
        Returns:
            str: Cleaned value
        """
        if not value or value.strip() == "":
            return "0"
        
        value = value.strip()
        
        # If value is just a dash or hyphen, replace with 0
        if value in ["-", "—", "–", "N/A", "n/a", "NA", "na"]:
            return "0"
        
        # Check if value is in parentheses (negative number)
        is_negative = False
        if value.startswith("(") and value.endswith(")"):
            is_negative = True
            value = value[1:-1].strip()
        
        # Remove currency symbols and commas
        cleaned = value.replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
        cleaned = cleaned.replace(",", "").replace(" ", "")
        
        # Handle percentage - keep the % sign if present
        # Remove other non-numeric characters except . and %
        if "%" in cleaned:
            # Keep numbers, dot, and %
            cleaned = re.sub(r'[^\d.%]', '', cleaned)
        else:
            # Keep only numbers and dot
            cleaned = re.sub(r'[^\d.]', '', cleaned)
        
        # If nothing left or empty, return 0
        if not cleaned or cleaned == ".":
            return "0"
        
        # Apply negative sign if it was in parentheses
        if is_negative and cleaned != "0":
            cleaned = "-" + cleaned
        
        return cleaned

    @staticmethod
    def is_row_empty(row: list, skip_first_column: bool = True) -> bool:
        """
        Check if a row is empty (all value columns are None, empty, or whitespace).
        
        Args:
            row (list): The row to check
            skip_first_column (bool): If True, skip the first column (row header) when checking
            
        Returns:
            bool: True if all value columns are empty, False otherwise
        """
        if not row:
            return True
        
        # Start from index 1 if skipping first column (row header), else start from 0
        start_idx = 1 if skip_first_column else 0
        
        # Check if all columns (after the first) are empty
        for col_idx in range(start_idx, len(row)):
            value = row[col_idx]
            # Check if value is not None and not empty/whitespace
            if value is not None and str(value).strip() != "":
                return False
        
        return True

    def clean_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a table by:
        1. Removing rows where all value columns (excluding first column) are empty/None
        2. Applying numeric cleaning to remaining value columns (all except the first)
        
        The first column is treated as row headers and kept as-is.
        
        Args:
            table (dict): Table dictionary with 'headers', 'rows', etc.
            
        Returns:
            dict: New table dictionary with cleaned data and original data preserved
        """
        if not table or "rows" not in table or not table["rows"]:
            return table
        
        cleaned_table = table.copy()
        original_rows = table["rows"]
        
        # Step 1: Remove rows where all value columns are empty
        non_empty_rows = []
        removed_rows_count = 0
        
        for row in original_rows:
            if not row:
                continue
            
            # Check if all value columns (except first) are empty
            if self.is_row_empty(row, skip_first_column=True):
                removed_rows_count += 1
                continue  # Skip this row
            
            non_empty_rows.append(row)
        
        if removed_rows_count > 0:
            logging.info(f"  ℹ️  Removed {removed_rows_count} empty row(s) from {table.get('table_name', 'table')}")
        
        # Step 2: Apply cleaning to non-empty rows
        cleaned_rows = []
        
        for row in non_empty_rows:
            cleaned_row = []
            for col_idx, value in enumerate(row):
                # First column is row header - keep as-is
                if col_idx == 0:
                    cleaned_row.append(value)
                else:
                    # Clean numeric columns
                    cleaned_value = self.clean_numeric_value(value)
                    cleaned_row.append(cleaned_value)
            
            cleaned_rows.append(cleaned_row)
        
        # Add both original and cleaned data
        cleaned_table["rows_original"] = original_rows
        cleaned_table["rows"] = cleaned_rows
        
        return cleaned_table

    def clean_all_tables(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean all tables in the extraction result.
        
        Args:
            extraction_result (dict): Result from analyze_with_vision with 'tables' key
            
        Returns:
            dict: New extraction result with cleaned tables
        """
        if not extraction_result or "tables" not in extraction_result:
            return extraction_result
        
        cleaned_result = {"tables": []}
        
        for table in extraction_result["tables"]:
            cleaned_table = self.clean_table(table)
            cleaned_result["tables"].append(cleaned_table)
        
        return cleaned_result