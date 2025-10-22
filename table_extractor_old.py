import os
from typing import Optional, Dict, Any, List
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import json
import re
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from azure_client import get_azure_openai_client
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class PDFTableExtractor:
    def __init__(self):
        # Initialize Azure OpenAI client
        self.client: AzureChatOpenAI = get_azure_openai_client()
        if self.client is None:
            raise ValueError("Failed to initialize Azure OpenAI client")

    def convert_pdf_to_base64(self, pdf_path: str) -> list:
        """
        Convert all pages of a PDF to base64 encoded images using PyMuPDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of base64 encoded images for each page
        """
        try:
            # Open the PDF file
            pdf_document = fitz.open(pdf_path)
            base64_images = []
            
            for page in pdf_document:
                # Get the page as an image with higher resolution
                zoom = 2  # Increase zoom for better quality
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert pixmap to bytes in PNG format
                img_data = pix.tobytes("png")
                
                # Convert directly to base64
                img_str = base64.b64encode(img_data).decode()
                base64_images.append(img_str)
            
            pdf_document.close()
            return base64_images
        except Exception as e:
            raise Exception(f"Error converting PDF to base64: {str(e)}")

    def analyze_with_vision(self, image_base64: str) -> dict:
        """
        Analyze the image using Azure OpenAI's vision model to identify and structure tables.
        """
        try:
            system_prompt = """You are a table extraction specialist. Extract tables from images and return ONLY valid JSON.

CRITICAL: You MUST return ONLY a valid JSON object, no markdown, no explanations, no code blocks.

TABLE EXTRACTION RULES:
1. **Column alignment is critical**: Each value must align with its column header.
2. **Row alignment is critical**: Each value must align with its row header
3. **Preserve all data exactly**: Do not skip, merge, or modify cell values


Output Format (STRICT JSON ONLY):
{
  "tables": [
    {
      "table_name": "Name or title of the table",
      "metadata": "Short description of table type (e.g., 'Balance Sheet', 'Income Statement', 'Cash Flow', 'Financial Summary', 'Sales Data', 'Expense Report', etc.)",
      "headers": ["header1", "header2", "header3", "..."],
      "rows": [
        ["value1", "value2", "value3", "..."],
        ["value1", "value2", "value3", "..."],
        ["value1", "value2", "value3", "..."]
      ]
    }
  ]
}

IMPORTANT:
1. Include "metadata" field with a brief description of what type of table this is
2. Return ONLY the JSON object, nothing else
3. No markdown formatting, no code blocks, no explanations
4. Each row must have EXACTLY the same number of elements as the headers array
5. Empty cells should be represented as empty strings "", not omitted"""

            content = [
                {
                    "type": "text",
                    "text": """Extract tables from this image that have numbered titles (e.g., "Table 1:", "Table 2:", "Table 3: Summary of Financial Results") and return as STRICT JSON ONLY.

CRITICAL EXTRACTION INSTRUCTIONS:

**Common Mistakes to AVOID:**
❌ Skipping empty cells (always include as "")
❌ Merging adjacent cells into one value
❌ Misaligning values with wrong columns
❌ Omitting rows that appear blank
❌ Transposing rows and columns
❌ Including extra values not in the table

**Requirements:**
1. ONLY extract tables with title format "Table X:" or "Table X: Description" (X = number)
2. Extract COMPLETE table name including number and description
3. Extract ALL column headers in exact left-to-right order
4. Extract ALL data rows, ensuring each has same length as headers
5. Table titles usually appear ABOVE the table (centered or left-aligned)
6. Analyze content and provide metadata:
   - Examples: "Balance Sheet", "Income Statement", "Cash Flow", "Financial Summary", "Sales Report", etc.

**Return Format:**
{
  "tables": [
    {
      "table_name": "Table 3: Summary of Financial Results",
      "metadata": "Income Statement",
      "headers": ["Column1", "Column2", "Column3"],
      "rows": [
        ["row1_col1", "row1_col2", "row1_col3"],
        ["row2_col1", "row2_col2", "row2_col3"],
        ["row3_col1", "", "row3_col3"]
      ]
    }
  ]
}

Return ONLY the JSON object. No markdown, no code blocks, no extra text. Skip tables without numbered titles."""
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
                HumanMessage(content=content)
            ]

            # Get response from model
            response = self.client.invoke(messages,temperature=0)
            response_text = response.content.strip()
            
            print(f"Raw LLM Response: {response_text[:500]}...")  # Show first 500 chars
            
            try:
                # Clean response - remove markdown code blocks if present
                if response_text.startswith("```"):
                    # Remove markdown code blocks
                    lines = response_text.split('\n')
                    # Find first { and last }
                    json_lines = []
                    in_json = False
                    for line in lines:
                        if line.strip().startswith('{'):
                            in_json = True
                        if in_json:
                            json_lines.append(line)
                        if line.strip().endswith('}') and in_json:
                            break
                    response_text = '\n'.join(json_lines)
                
                # Parse JSON
                result = json.loads(response_text)
                
                # Validate structure
                if not isinstance(result, dict):
                    print("Response was not a dictionary")
                    result = {"tables": []}
                elif "tables" not in result:
                    print("No 'tables' key in response")
                    result = {"tables": []}
                
                print(f"✅ Successfully parsed JSON with {len(result.get('tables', []))} table(s)")
                return result
                
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                # Try to extract JSON from the response if it's embedded in text
                try:
                    # Look for JSON-like structure in the text
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = response_text[start_idx:end_idx]
                        result = json.loads(json_str)
                        print("✅ Extracted and parsed JSON from response")
                        return result
                    print("❌ Could not find valid JSON in response")
                    return {"tables": []}
                except Exception as json_extract_error:
                    print(f"Failed to extract JSON from response: {json_extract_error}")
                    return {"tables": []}
                    
        except Exception as e:
            print(f"Vision Analysis Error: {str(e)}")
            raise Exception(f"Error analyzing text with LLM: {str(e)}")

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
            print(f"  ℹ️  Removed {removed_rows_count} empty row(s) from {table.get('table_name', 'table')}")
        
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




