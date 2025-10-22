"""
Streamlit UI for Table Extraction and Validation

This application allows users to:
1. Upload a PDF document
2. Extract tables with metadata
3. Analyze tables for casting validation needs
4. View detailed validation results
"""

import streamlit as st
import json
import tempfile
import os
from table_extractor_old import PDFTableExtractor
from validation_agent import TableValidationAgent, ValidationReport


# Page configuration
st.set_page_config(
    page_title="Table Extraction & Validation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .validation-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .needs-validation {
        border-left-color: #ff7f0e !important;
    }
    .no-validation {
        border-left-color: #2ca02c !important;
    }
    </style>
""", unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with instructions and settings"""
    with st.sidebar:
        st.title("📋 Instructions")
        st.markdown("""
        ### How to Use:
        1. **Upload PDF**: Click "Browse files" to upload your PDF document
        2. **Wait for Processing**: The app will extract tables from the PDF
        3. **View Results**: See extracted tables and validation analysis
        4. **Download Reports**: Export results as JSON
        
        ### What is Casting Validation?
        Casting validation checks if table data matches expected types:
        - ✅ Numbers in numeric columns
        - ✅ Valid currency formatting
        - ✅ Proper date formats
        - ✅ Correct percentage values
        - ✅ Arithmetic accuracy in totals
        """)
        
        st.divider()
        
        st.markdown("### ⚙️ Settings")
        show_raw_json = st.checkbox("Show Raw JSON", value=False)
        show_extraction_details = st.checkbox("Show Extraction Details", value=False)
        
        return show_raw_json, show_extraction_details


def render_metrics(report: ValidationReport):
    """Render summary metrics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Total Tables",
            value=report.total_tables
        )
    
    with col2:
        st.metric(
            label="✅ Need Validation",
            value=report.tables_needing_validation
        )
    
    with col3:
        percentage = (report.tables_needing_validation / report.total_tables * 100) if report.total_tables > 0 else 0
        st.metric(
            label="📈 Validation Rate",
            value=f"{percentage:.1f}%"
        )


def render_table_card(table_data: dict, index: int, show_cleaned: bool = False):
    """Render a table extraction card"""
    with st.expander(f"📋 {table_data['table_name']}", expanded=False):
        st.markdown(f"**Type:** {table_data.get('metadata', 'N/A')}")
        
        # Display table headers
        if table_data.get('headers'):
            st.markdown("**Headers:**")
            st.code(", ".join(table_data['headers']))
        
        # Display row count
        row_count = len(table_data.get('rows', []))
        st.markdown(f"**Rows:** {row_count}")
        
        # Show all rows
        if table_data.get('rows'):
            import pandas as pd
            
            # Create tabs for original and cleaned if cleaned data exists
            if show_cleaned and table_data.get('rows_original'):
                tab1, tab2 = st.tabs(["🧹 Cleaned Data", "📄 Original Data"])
                
                with tab1:
                    st.markdown("**Cleaned Table Data:**")
                    if table_data.get('headers'):
                        df_data = []
                        for row in table_data['rows']:
                            if row:
                                df_data.append(dict(zip(table_data['headers'], row)))
                        if df_data:
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True)
                
                with tab2:
                    st.markdown("**Original Table Data:**")
                    if table_data.get('headers'):
                        df_data = []
                        for row in table_data['rows_original']:
                            if row:
                                df_data.append(dict(zip(table_data['headers'], row)))
                        if df_data:
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True)
            else:
                # No cleaning - just show the data
                st.markdown("**Table Data:**")
                if table_data.get('headers'):
                    df_data = []
                    for row in table_data['rows']:
                        if row:
                            df_data.append(dict(zip(table_data['headers'], row)))
                    if df_data:
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)


def render_validation_analysis(analysis, index: int, cleaned_result: dict):
    """Render validation analysis for a single table"""
    
    # Validation status emoji
    status_emoji = "✅" if analysis.needs_casting_validation else "❌"
    status_text = "YES" if analysis.needs_casting_validation else "NO"
    
    # Create expandable card
    with st.expander(
        f"{status_emoji} {analysis.table_name}", 
        expanded=analysis.needs_casting_validation
    ):
        # Header info
        st.markdown(f"**Needs Casting Validation:** `{status_text}`")
        
        st.divider()
        
        # Explanation
        st.markdown("#### 📝 Reasoning")
        st.info(analysis.explanation)
        
        st.divider()
        
        # Show extracted table data
        st.markdown("#### 📋 Table Data")
        
        # Find the corresponding table in cleaned_result
        table_data = None
        for table in cleaned_result.get('tables', []):
            if table['table_name'] == analysis.table_name:
                table_data = table
                break
        
        if table_data:
            # Display table metadata
            if table_data.get('metadata'):
                st.markdown(f"**Type:** {table_data['metadata']}")
            
            # Display headers
            if table_data.get('headers'):
                st.markdown(f"**Columns:** {', '.join(table_data['headers'])}")
            
            # Display table as dataframe with tabs for original and cleaned
            if table_data.get('headers') and table_data.get('rows'):
                import pandas as pd
                
                # Create tabs for cleaned and original data
                if table_data.get('rows_original'):
                    tab1, tab2 = st.tabs(["🧹 Cleaned Data", "📄 Original Data"])
                    
                    with tab1:
                        # Create dataframe from cleaned rows
                        df_data = []
                        for row in table_data['rows']:
                            if row and len(row) == len(table_data['headers']):
                                df_data.append(dict(zip(table_data['headers'], row)))
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True, height=min(len(df_data) * 35 + 38, 400))
                        else:
                            st.warning("No valid table data to display")
                    
                    with tab2:
                        # Create dataframe from original rows
                        df_data = []
                        for row in table_data['rows_original']:
                            if row and len(row) == len(table_data['headers']):
                                df_data.append(dict(zip(table_data['headers'], row)))
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True, height=min(len(df_data) * 35 + 38, 400))
                        else:
                            st.warning("No valid original data to display")
                else:
                    # No original data - just show current data
                    df_data = []
                    for row in table_data['rows']:
                        if row and len(row) == len(table_data['headers']):
                            df_data.append(dict(zip(table_data['headers'], row)))
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True, height=min(len(df_data) * 35 + 38, 400))
                    else:
                        st.warning("No valid table data to display")
            
            # Show row count
            row_count = len(table_data.get('rows', []))
            st.caption(f"📊 Total rows: {row_count}")
        else:
            st.warning("Table data not found in extraction results")


def render_summary_section(report: ValidationReport):
    """Render overall summary"""
    st.markdown("### 📊 Summary")
    st.info(report.summary)
    
    # Tables needing validation
    tables_needing = [a for a in report.analyses if a.needs_casting_validation]
    tables_not_needing = [a for a in report.analyses if not a.needs_casting_validation]
    
    if tables_needing:
        st.warning(f"**✅ Tables Requiring Validation ({len(tables_needing)}):**")
        for analysis in tables_needing:
            st.markdown(f"- {analysis.table_name}")
    
    if tables_not_needing:
        st.success(f"**❌ Tables NOT Requiring Validation ({len(tables_not_needing)}):**")
        for analysis in tables_not_needing:
            st.markdown(f"- {analysis.table_name}")


def process_pdf(pdf_file, show_extraction_details: bool):
    """Process PDF and return extraction and validation results"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Step 1: Extract tables
        with st.spinner("🔍 Extracting tables from PDF..."):
            extractor = PDFTableExtractor()
            
            # Convert PDF to base64 images
            base64_images = extractor.convert_pdf_to_base64(tmp_path)
            st.success(f"✅ Converted {len(base64_images)} page(s) to images")
            
            # Extract tables from each page
            all_tables = []
            progress_bar = st.progress(0)
            
            for i, image_base64 in enumerate(base64_images):
                if show_extraction_details:
                    st.info(f"📄 Analyzing page {i+1}/{len(base64_images)}...")
                
                result = extractor.analyze_with_vision(image_base64)
                
                if result and "tables" in result:
                    page_tables = result["tables"]
                    all_tables.extend(page_tables)
                    
                    if show_extraction_details:
                        st.success(f"✅ Found {len(page_tables)} table(s) on page {i+1}")
                
                progress_bar.progress((i + 1) / len(base64_images))
            
            extraction_result = {"tables": all_tables}
            
            if len(all_tables) == 0:
                st.warning("⚠️ No tables found in the PDF")
                return None, None, None
            
            st.success(f"✅ Extracted {len(all_tables)} table(s) total")
        
        # Step 2: Clean tables
        with st.spinner("🧹 Cleaning table data..."):
            cleaned_result = extractor.clean_all_tables(extraction_result)
            st.success("✅ Table cleaning complete")
        
        # Step 3: Validate tables
        with st.spinner("🤖 Analyzing tables with AI agent..."):
            agent = TableValidationAgent()
            validation_report = agent.analyze_from_dict(cleaned_result)
            st.success("✅ Validation analysis complete")
        
        return extraction_result, cleaned_result, validation_report
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        import traceback
        with st.expander("Show Error Details"):
            st.code(traceback.format_exc())
        return None, None, None
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def main():
    """Main application"""
    
    # Header
    st.title("📊 Table Extraction & Validation System")
    st.markdown("""
    Upload a PDF document to extract tables and analyze if they need data type casting validation.
    """)
    
    # Sidebar
    show_raw_json, show_extraction_details = render_sidebar()
    
    st.divider()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF file containing tables to analyze"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.markdown(f"**📄 File:** {uploaded_file.name}")
        st.markdown(f"**📦 Size:** {uploaded_file.size / 1024:.2f} KB")
        
        # Process button
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            # Process the PDF
            extraction_result, cleaned_result, validation_report = process_pdf(uploaded_file, show_extraction_details)
            
            if extraction_result and cleaned_result and validation_report:
                # Store in session state
                st.session_state['extraction_result'] = extraction_result
                st.session_state['cleaned_result'] = cleaned_result
                st.session_state['validation_report'] = validation_report
    
    # Display results if available
    if 'validation_report' in st.session_state and 'extraction_result' in st.session_state and 'cleaned_result' in st.session_state:
        st.divider()
        
        validation_report = st.session_state['validation_report']
        extraction_result = st.session_state['extraction_result']
        cleaned_result = st.session_state['cleaned_result']
        
        # Metrics
        st.markdown("## 📈 Overview")
        render_metrics(validation_report)
        
        st.divider()
        
        # Summary
        render_summary_section(validation_report)
        
        st.divider()
        
        # Detailed Results
        st.markdown("## 🔍 Detailed Analysis")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Validation Analysis", "Cleaned Tables", "Original Tables", "Export"])
        
        with tab1:
            st.markdown("### Validation Results")
            for i, analysis in enumerate(validation_report.analyses):
                render_validation_analysis(analysis, i, cleaned_result)
        
        with tab2:
            st.markdown("### Cleaned Tables")
            st.info("📊 Tables with numeric values cleaned (currency symbols removed, parentheses converted to negatives, empty cells = 0)")
            for i, table in enumerate(cleaned_result['tables']):
                render_table_card(table, i, show_cleaned=True)
        
        with tab3:
            st.markdown("### Original Extracted Tables")
            st.info("📄 Raw tables as extracted from the PDF without any cleaning")
            for i, table in enumerate(extraction_result['tables']):
                render_table_card(table, i, show_cleaned=False)
        
        with tab4:
            st.markdown("### 📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Validation Report")
                report_json = json.dumps(
                    validation_report.model_dump(), 
                    indent=2,
                    ensure_ascii=False
                )
                st.download_button(
                    label="📥 Download Validation Report",
                    data=report_json,
                    file_name="validation_report.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                if show_raw_json:
                    with st.expander("Preview JSON"):
                        st.json(validation_report.model_dump())
            
            with col2:
                st.markdown("#### Cleaned Tables")
                cleaned_json = json.dumps(
                    cleaned_result,
                    indent=2,
                    ensure_ascii=False
                )
                st.download_button(
                    label="📥 Download Cleaned Tables",
                    data=cleaned_json,
                    file_name="cleaned_tables.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                if show_raw_json:
                    with st.expander("Preview JSON"):
                        st.json(cleaned_result)
            
            with col3:
                st.markdown("#### Original Tables")
                extraction_json = json.dumps(
                    extraction_result,
                    indent=2,
                    ensure_ascii=False
                )
                st.download_button(
                    label="📥 Download Original Tables",
                    data=extraction_json,
                    file_name="original_tables.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                if show_raw_json:
                    with st.expander("Preview JSON"):
                        st.json(extraction_result)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Powered by Azure OpenAI (GPT-4o)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
