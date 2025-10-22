"""
Pydantic AI Agent for Table Validation Analysis

This module uses Pydantic data models and Azure OpenAI to analyze extracted tables
and determine if they need casting validation (data type verification).
"""

import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from azure_client import get_azure_openai_client


# ==================== Pydantic Data Models ====================

class ExtractedTable(BaseModel):
    """Represents an extracted table from the table_extractor"""
    table_name: str = Field(..., description="Name or title of the table")
    metadata: str = Field(..., description="Short description of table type")
    headers: List[str] = Field(..., description="Column headers")
    rows: List[List[str]] = Field(..., description="Table rows with data values")


class TableExtractionResult(BaseModel):
    """Complete result from table extraction"""
    tables: List[ExtractedTable] = Field(default_factory=list, description="List of extracted tables")


class ValidationAnalysis(BaseModel):
    """Analysis result for a single table"""
    table_name: str = Field(..., description="Name of the analyzed table")
    metadata: str = Field(..., description="Table type/category")
    needs_casting_validation: bool = Field(
        ..., 
        description="Whether the table needs data type casting validation"
    )
    explanation: str = Field(
        ..., 
        description="Detailed explanation of why casting validation is or isn't needed"
    )
    detected_data_types: Dict[str, str] = Field(
        default_factory=dict,
        description="Expected data types for each column"
    )
    risk_level: str = Field(
        ..., 
        description="Risk level: low, medium, or high",
        pattern="^(low|medium|high)$"
    )


class ValidationReport(BaseModel):
    """Complete validation report for all tables"""
    total_tables: int = Field(..., description="Total number of tables analyzed")
    tables_needing_validation: int = Field(..., description="Number of tables requiring casting validation")
    analyses: List[ValidationAnalysis] = Field(..., description="Individual analysis for each table")
    summary: str = Field(..., description="Overall summary of the validation report")


# ==================== Pydantic AI Agent ====================

class TableValidationAgent:
    """
    AI Agent that analyzes extracted tables and determines if they need casting validation.
    
    Casting validation checks if:
    - Numeric columns contain valid numbers
    - Date columns contain valid dates
    - Currency columns are properly formatted
    - Percentages are valid
    - Text columns don't contain unexpected numeric data
    """
    
    def __init__(self):
        """Initialize the agent with Azure OpenAI client"""
        self.client: AzureChatOpenAI = get_azure_openai_client()
        if self.client is None:
            raise ValueError("Failed to initialize Azure OpenAI client")
    
    def analyze_table(self, table: ExtractedTable) -> ValidationAnalysis:
        """
        Analyze a single table and determine if it needs casting validation.
        
        Args:
            table: ExtractedTable object containing table data
            
        Returns:
            ValidationAnalysis with detailed analysis results
        """
        try:
            # Prepare table data for analysis
            table_data = {
                "table_name": table.table_name,
                "metadata": table.metadata,
                "headers": table.headers,
                "sample_rows": table.rows[:10]  # Send first 10 rows for analysis
            }
            
            system_prompt = """You are a data quality expert specializing in table validation and data type casting.

Your task is to analyze tables and determine if they need CASTING VALIDATION.

CASTING VALIDATION is needed when:
1. **Numeric columns** might contain non-numeric values (text, symbols, formatting issues)
2. **Financial data** (currency, amounts, percentages) that could have formatting inconsistencies
3. **Date/Time columns** that might have invalid or inconsistent date formats
4. **Mixed data types** in columns that should be homogeneous
5. **Calculated fields** (totals, sub-totals) that need arithmetic verification
6. **Data imported from external sources** prone to type errors

CASTING VALIDATION is NOT needed when:
1. **Pure text tables** (names, descriptions, categorical data only)
2. **Already validated data** from trusted sources
3. **Simple reference tables** (lookups, mappings)
4. **Tables with minimal numeric data** and no calculations

Analyze the table structure, headers, and data patterns to make your determination.

You MUST respond with ONLY a valid JSON object matching this exact structure:
{
  "table_name": "string",
  "metadata": "string",
  "needs_casting_validation": true/false,
  "explanation": "Detailed explanation of why casting validation is or isn't needed. Mention specific columns and data types.",
  "detected_data_types": {
    "column_name": "data_type (e.g., integer, float, currency, percentage, date, text, mixed)"
  },
  "risk_level": "low/medium/high"
}

Return ONLY the JSON object. No markdown, no code blocks, no extra text."""

            user_prompt = f"""Analyze this table and determine if it needs casting validation:

Table Data:
{json.dumps(table_data, indent=2)}

Consider:
1. What data types are in each column?
2. Are there numeric/financial columns that could have type errors?
3. Are there calculations (totals, sub-totals) that need validation?
4. What's the risk level if data types are incorrect?
5. Is this table prone to data type casting errors?

Return your analysis as a JSON object following the specified format."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Get response from model
            response = self.client.invoke(messages)
            response_text = response.content.strip()
            
            # Clean response - remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split('\n')
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
            
            # Parse JSON and create ValidationAnalysis object
            analysis_data = json.loads(response_text)
            return ValidationAnalysis(**analysis_data)
            
        except Exception as e:
            # Return a default analysis if something goes wrong
            print(f"Error analyzing table {table.table_name}: {str(e)}")
            return ValidationAnalysis(
                table_name=table.table_name,
                metadata=table.metadata,
                needs_casting_validation=True,  # Default to true for safety
                explanation=f"Error during analysis: {str(e)}. Recommending validation as a precaution.",
                detected_data_types={},
                risk_level="medium"
            )
    
    def analyze_all_tables(self, extraction_result: TableExtractionResult) -> ValidationReport:
        """
        Analyze all tables from the extraction result.
        
        Args:
            extraction_result: TableExtractionResult containing all extracted tables
            
        Returns:
            ValidationReport with complete analysis for all tables
        """
        analyses = []
        
        for table in extraction_result.tables:
            analysis = self.analyze_table(table)
            analyses.append(analysis)
        
        # Calculate summary statistics
        total_tables = len(analyses)
        tables_needing_validation = sum(1 for a in analyses if a.needs_casting_validation)
        
        # Generate overall summary
        summary = self._generate_summary(analyses, total_tables, tables_needing_validation)
        
        return ValidationReport(
            total_tables=total_tables,
            tables_needing_validation=tables_needing_validation,
            analyses=analyses,
            summary=summary
        )
    
    def _generate_summary(self, analyses: List[ValidationAnalysis], 
                         total_tables: int, tables_needing_validation: int) -> str:
        """Generate a summary of the validation analysis"""
        if total_tables == 0:
            return "No tables were analyzed."
        
        percentage = (tables_needing_validation / total_tables) * 100
        
        high_risk = sum(1 for a in analyses if a.risk_level == "high")
        medium_risk = sum(1 for a in analyses if a.risk_level == "medium")
        low_risk = sum(1 for a in analyses if a.risk_level == "low")
        
        summary_parts = [
            f"Analyzed {total_tables} table(s).",
            f"{tables_needing_validation} table(s) ({percentage:.1f}%) require casting validation."
        ]
        
        if high_risk > 0:
            summary_parts.append(f"{high_risk} table(s) have HIGH risk level.")
        if medium_risk > 0:
            summary_parts.append(f"{medium_risk} table(s) have MEDIUM risk level.")
        if low_risk > 0:
            summary_parts.append(f"{low_risk} table(s) have LOW risk level.")
        
        if tables_needing_validation > 0:
            summary_parts.append("Recommend implementing data type validation before processing.")
        else:
            summary_parts.append("All tables appear to be low risk for data type issues.")
        
        return " ".join(summary_parts)
    
    def analyze_from_dict(self, extraction_dict: Dict[str, Any]) -> ValidationReport:
        """
        Convenience method to analyze tables from a dictionary (direct output from table_extractor).
        
        Args:
            extraction_dict: Dictionary with 'tables' key containing table data
            
        Returns:
            ValidationReport with complete analysis
        """
        extraction_result = TableExtractionResult(**extraction_dict)
        return self.analyze_all_tables(extraction_result)


# ==================== Helper Functions ====================

def print_validation_report(report: ValidationReport, detailed: bool = True):
    """
    Pretty print the validation report.
    
    Args:
        report: ValidationReport to print
        detailed: If True, print detailed analysis for each table
    """
    print("\n" + "="*80)
    print("TABLE VALIDATION REPORT")
    print("="*80)
    print(f"\n📊 Summary: {report.summary}\n")
    
    if detailed:
        for i, analysis in enumerate(report.analyses, 1):
            print(f"\n{'─'*80}")
            print(f"Table {i}: {analysis.table_name}")
            print(f"{'─'*80}")
            print(f"Type: {analysis.metadata}")
            print(f"Needs Validation: {'✓ YES' if analysis.needs_casting_validation else '✗ NO'}")
            print(f"Risk Level: {analysis.risk_level.upper()}")
            print("\nExplanation:")
            print(f"  {analysis.explanation}")
            
            if analysis.detected_data_types:
                print("\nDetected Data Types:")
                for col, dtype in analysis.detected_data_types.items():
                    print(f"  • {col}: {dtype}")

