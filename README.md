# 📊 Table Extraction & Validation System

A complete AI-powered solution for extracting tables from PDF documents and analyzing them for data type casting validation needs.

## 🎯 Overview

This system combines:
1. **PDF Table Extraction** - Extracts tables with metadata using Azure OpenAI Vision (GPT-4o)
2. **Pydantic AI Agent** - Analyzes tables and determines validation requirements
3. **Streamlit Web UI** - User-friendly interface for the entire workflow

## ✨ Features

### Table Extraction
- ✅ Extracts tables from PDF documents
- ✅ Identifies table names and types (metadata)
- ✅ Classifies row types (normal, sub-total, total)
- ✅ Preserves table structure and headers
- ✅ Processes multiple pages

### Validation Analysis
- ✅ AI-powered validation recommendation
- ✅ Boolean answer: Needs validation? (Yes/No)
- ✅ Detailed explanations for decisions
- ✅ Data type detection per column
- ✅ Risk level assessment (Low/Medium/High)
- ✅ Structured Pydantic models

### Web Interface
- ✅ Simple drag-and-drop PDF upload
- ✅ Real-time processing feedback
- ✅ Interactive results dashboard
- ✅ Visual metrics and summaries
- ✅ Color-coded risk indicators
- ✅ Export to JSON

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI
Create a `.env` file or set environment variables:
```env
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-10-01-preview
```

### 3. Run the Web UI
```powershell
# Option 1: Use the launch script
.\run_app.ps1

# Option 2: Direct command
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
table-extractor-3/
│
├── Core Components
│   ├── table_extractor.py       # PDF extraction engine
│   ├── validation_agent.py      # Pydantic AI validation agent
│   ├── azure_client.py          # Azure OpenAI client
│   └── streamlit_app.py         # Web UI ⭐ NEW!
│
├── Utilities & Examples
│   ├── example_usage.py         # Integration examples
│   ├── test_validation_agent.py # Test suite
│   └── run_app.ps1             # Launch script ⭐ NEW!
│
├── Documentation
│   ├── README.md                      # This file (master README)
│   ├── QUICKSTART.md                  # Quick start guide
│   ├── SOLUTION_SUMMARY.md            # Feature overview
│   ├── ARCHITECTURE.md                # System architecture
│   ├── VALIDATION_AGENT_README.md     # Agent documentation
│   ├── STREAMLIT_README.md            # UI documentation ⭐ NEW!
│   ├── STREAMLIT_SUMMARY.md           # UI summary ⭐ NEW!
│   └── UI_GUIDE.md                    # Visual UI guide ⭐ NEW!
│
└── Configuration
    └── requirements.txt           # Python dependencies
```

## 💡 Usage Examples

### Using the Web UI (Recommended)

1. Run: `streamlit run streamlit_app.py`
2. Upload your PDF
3. Click "Process Document"
4. View results and download JSON

### Using Python API

```python
from table_extractor import PDFTableExtractor
from validation_agent import TableValidationAgent

# Extract tables
extractor = PDFTableExtractor()
images = extractor.convert_pdf_to_base64("document.pdf")
tables = []
for img in images:
    result = extractor.analyze_with_vision(img)
    tables.extend(result["tables"])

# Analyze with validation agent
agent = TableValidationAgent()
report = agent.analyze_from_dict({"tables": tables})

# Access results
for analysis in report.analyses:
    print(f"Table: {analysis.table_name}")
    print(f"Needs Validation: {analysis.needs_casting_validation}")
    print(f"Explanation: {analysis.explanation}")
    print(f"Risk: {analysis.risk_level}")
```

## 📊 Output Examples

### Web UI Output
- Interactive dashboard with metrics
- Color-coded validation cards
- Detailed analysis per table
- Downloadable JSON reports

### JSON Output
```json
{
  "table_name": "Table 1: Financial Summary",
  "metadata": "Income Statement",
  "needs_casting_validation": true,
  "explanation": "This financial table contains currency values and percentages requiring type validation...",
  "detected_data_types": {
    "Revenue": "currency",
    "Expenses": "currency",
    "Growth": "percentage"
  },
  "risk_level": "high"
}
```

## 🎯 What is Casting Validation?

Casting validation ensures data in table columns matches expected types:
- **Numeric columns** contain valid numbers (not text)
- **Currency columns** have proper formatting
- **Date columns** contain valid dates
- **Percentage columns** are correctly formatted
- **Calculated totals** are arithmetically accurate

## 🎨 Key Components

### 1. Table Extractor (`table_extractor.py`)
- Converts PDFs to images
- Uses Azure OpenAI Vision (GPT-4o)
- Extracts structured table data
- Generates metadata descriptions

### 2. Validation Agent (`validation_agent.py`)
- Pydantic-based AI agent
- Analyzes table structure and content
- Determines validation requirements
- Returns structured responses

### 3. Streamlit UI (`streamlit_app.py`)
- Web-based interface
- Upload and process PDFs
- View interactive results
- Export JSON reports

## 📦 Requirements

- Python 3.8+
- Azure OpenAI access with GPT-4o
- Internet connection
- Valid Azure credentials

### Python Packages
- streamlit
- langchain-openai
- langchain-core
- PyMuPDF
- pydantic >= 2.0.0
- pandas
- azure-identity
- azure-core

## 🔧 Configuration

### Environment Variables
```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-10-01-preview
AZURE_OPENAI_SCOPE=https://cognitiveservices.azure.com/.default
```

### UI Settings (in sidebar)
- Show Raw JSON: Preview JSON before download
- Show Extraction Details: Verbose extraction logs

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **README.md** | Master documentation (this file) |
| **QUICKSTART.md** | 5-minute getting started guide |
| **STREAMLIT_README.md** | Complete UI documentation |
| **VALIDATION_AGENT_README.md** | Agent API reference |
| **ARCHITECTURE.md** | System design and flow |
| **UI_GUIDE.md** | Visual UI walkthrough |

## 🧪 Testing

### Test Validation Agent
```powershell
python test_validation_agent.py
```

### Test with Sample Data
```powershell
python example_usage.py --sample
```

### Test Web UI
```powershell
streamlit run streamlit_app.py
# Upload a PDF and test the workflow
```

## 🎯 Use Cases

### Financial Analysis
- Extract financial statements
- Validate currency and percentage data
- Verify calculations in totals

### Data Quality
- Identify tables needing validation
- Prioritize by risk level
- Ensure data type consistency

### Document Processing
- Batch process multiple PDFs
- Automated table extraction
- Structured data export

### Compliance & Auditing
- Verify data accuracy
- Track validation requirements
- Export audit trails

## 🚨 Troubleshooting

### Issue: Azure authentication fails
**Solution:** Check environment variables and Azure permissions

### Issue: No tables found
**Solution:** Ensure tables have numbered titles (e.g., "Table 1:")

### Issue: Processing is slow
**Solution:** Normal for large PDFs; check progress indicators

### Issue: Import errors
**Solution:** Run `pip install -r requirements.txt`

For detailed troubleshooting, see `STREAMLIT_README.md`

## 🎓 Learning Path

1. **Start Here**: Read this README
2. **Quick Test**: Run `python test_validation_agent.py`
3. **Try UI**: Run `streamlit run streamlit_app.py`
4. **Explore Code**: Check `example_usage.py`
5. **Deep Dive**: Read component-specific docs

## 📈 Performance

| PDF Size | Processing Time |
|----------|----------------|
| 1-5 pages | 30-60 seconds |
| 5-15 pages | 2-4 minutes |
| 15+ pages | 5+ minutes |

*Time depends on: number of pages, tables per page, network speed*

## 🔒 Security

- PDFs temporarily stored during processing
- Automatic cleanup after processing
- Secure Azure OpenAI endpoints
- Azure Identity authentication
- No permanent data storage

## 🎉 What's New

### Latest Updates
- ✅ Added Streamlit web UI
- ✅ Interactive validation dashboard
- ✅ Visual risk indicators
- ✅ JSON export functionality
- ✅ Complete documentation suite

## 🤝 Components Overview

```
┌─────────────┐
│   PDF Doc   │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Table Extractor     │ ─── Extracts tables with metadata
│ (table_extractor.py)│
└──────┬──────────────┘
       │ Tables JSON
       ▼
┌─────────────────────┐
│ Validation Agent    │ ─── Analyzes validation needs
│ (validation_agent.py)│
└──────┬──────────────┘
       │ Validation Report
       ▼
┌─────────────────────┐
│ Streamlit UI        │ ─── Displays results
│ (streamlit_app.py)  │
└─────────────────────┘
```

## 💼 Professional Features

- ✅ Type-safe Pydantic models
- ✅ Comprehensive error handling
- ✅ Progress feedback
- ✅ Structured logging
- ✅ Export capabilities
- ✅ Modular architecture
- ✅ Well-documented code

## 🎯 Success Criteria

Your system is working correctly when:
- ✅ PDFs upload successfully
- ✅ Tables are extracted with metadata
- ✅ Validation analysis completes
- ✅ Results display in UI
- ✅ JSON exports successfully
- ✅ Risk levels are accurate
- ✅ Explanations are clear

## 📞 Support & Resources

- **Quick Start**: See `QUICKSTART.md`
- **UI Help**: See `STREAMLIT_README.md`
- **API Docs**: See `VALIDATION_AGENT_README.md`
- **Architecture**: See `ARCHITECTURE.md`

## 🎊 Getting Started in 3 Steps

```powershell
# 1. Install
pip install -r requirements.txt

# 2. Configure (set Azure environment variables)
# See Configuration section above

# 3. Run
streamlit run streamlit_app.py
```

## 🌟 Highlights

### For End Users
- Simple web interface
- No coding required
- Visual feedback
- Clear results
- Easy exports

### For Developers
- Clean architecture
- Pydantic type safety
- Comprehensive docs
- Extensible design
- Production-ready

### For Organizations
- Streamlined workflow
- Quality assurance
- Risk assessment
- Audit trails
- Time savings

---

## 🚀 Ready to Start?

```powershell
# Launch the app
streamlit run streamlit_app.py
```

**Then upload a PDF and see the magic happen! ✨**

---

**Built with ❤️ using:**
- 🤖 Azure OpenAI (GPT-4o)
- ✅ Pydantic
- 🎨 Streamlit
- 📊 Pandas
- 🐍 Python

**Project Status:** ✅ Production Ready

**License:** Part of table-extractor project

**Last Updated:** October 15, 2025
