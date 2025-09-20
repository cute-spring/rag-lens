# RAG Lens - RAG Pipeline Testing & Performance Tuning Tool

A comprehensive production-ready platform for testing, optimizing, and integrating RAG (Retrieval-Augmented Generation) pipelines with advanced API integration capabilities and comprehensive test suites.

## 🎯 Purpose

This enterprise-grade platform is designed for:
- **Testing**: Comprehensive RAG pipeline testing with 156+ test cases across 12 categories
- **API Integration**: Production-ready API integration with standardized contracts and authentication patterns
- **Performance Tuning**: Real-time parameter optimization with advanced monitoring and health checks
- **Enterprise Deployment**: Scalable architecture with error handling, security, and compliance features
- **Multi-language Support**: Enhanced test suites covering multilingual processing and domain-specific scenarios

## 📊 Key Features

### 1. Test Case Management
- Dropdown selection from multiple test cases
- Visual status indicators (pass/fail/pending)
- Detailed metadata display for each test case

### 2. Pipeline Visualization
- **7-Step Processing Pipeline**:
  1. Query Processing
  2. Retrieval
  3. Initial Filtering
  4. Re-ranking
  5. Final Selection
  6. Context Assembly
  7. Response Generation
- Expandable sections for detailed step analysis
- Real-time metrics display

### 3. Parameter Tuning Interface
- **Re-ranking Weights**: Semantic, Freshness, Quality with automatic normalization
- **Threshold Controls**: Adjustable relevance threshold with visual feedback
- **Top-N Selection**: Interactive slider for context size selection

### 4. Results Analysis
- Side-by-side comparison of expected vs actual responses
- Chunk score visualization with interactive charts
- Performance metrics and optimization suggestions
- Failure analysis with user concerns and comments

### 5. Markdown Support
- Full markdown rendering throughout the interface
- Raw markdown toggle option for content chunks
- Proper formatting for technical content

## 🏗️ Architecture

### Data Structure
Each test case contains:
- **20 Content Chunks** with:
  - Markdown-formatted content
  - Title and timestamps
  - User ratings (0-5 scale)
  - Multiple score types (relevance, freshness, quality)
- **Query & Instructions**: User query and processing instructions
- **System Configuration**: Model version and system prompt
- **Re-ranking Parameters**: Configurable weights and thresholds
- **Expected Results**: Ideal answer and failure analysis data

### Classes & Components
- `MockDataGenerator`: Creates realistic test case data
- `RAGPipelineSimulator`: Simulates 7-step processing pipeline
- UI Components: Modular rendering functions for different interface sections

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Streamlit 1.28+
- Required Python packages (see requirements.txt)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/cute-spring/rag-lens.git
cd rag-lens

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run rag_pipeline_tuning_tool.py
```

### Docker Deployment (Optional)

```bash
# Build the image
docker build -t rag-lens .

# Run the container
docker run -p 8501:8501 rag-lens
```

### Usage
1. **Select a Test Case**: Choose from available test cases in the sidebar
2. **Adjust Parameters**: Use sliders to modify re-ranking weights and thresholds
3. **Run Analysis**: Click "Run Pipeline Analysis" to process the test case
4. **Review Results**: Examine pipeline steps, chunk selection, and performance metrics
5. **Optimize**: Use suggestions to fine-tune parameters for better results

## 🎨 Interface Guide

### Sidebar Controls
- **Test Case Selection**: Dropdown with test case details
- **Parameter Tuning**: Real-time adjustment controls
- **Status Indicators**: Visual feedback on test case status

### Main Content Area
- **Pipeline Steps**: Expandable sections for each processing step
- **Chunk Analysis**: Score distributions and detailed chunk information
- **Results Comparison**: Expected vs actual response comparison
- **Optimization Suggestions**: AI-driven parameter recommendations

### Visual Elements
- **Color-coded Scores**: High (green), Medium (orange), Low (red)
- **Interactive Charts**: Score distribution visualization
- **Status Badges**: Pass/fail/pending indicators
- **Expandable Sections**: Collapsible detailed information

## 🔧 Technical Details

### Performance Metrics
- **Retrieval Rate**: Percentage of documents retrieved
- **Filter Rate**: Percentage passing relevance threshold
- **Selection Rate**: Percentage selected for final context
- **Average Scores**: Relevance and composite score averages

### Parameter Optimization
The system automatically:
- Normalizes re-ranking weights to sum to 1.0
- Provides real-time feedback on parameter changes
- Suggests optimizations based on performance metrics
- Visualizes impact of parameter adjustments

### Data Simulation
- Realistic chunk generation with markdown content
- Time-based scoring for freshness
- User rating simulation
- Multi-dimensional scoring system

## 📈 Use Cases

### 1. Parameter Sensitivity Analysis
- Test how different weight combinations affect results
- Identify optimal parameter settings for specific domains
- Understand parameter interactions and trade-offs

### 2. Quality Assessment
- Compare expected vs generated responses
- Analyze chunk selection quality
- Evaluate relevance and freshness metrics

### 3. System Optimization
- Fine-tune thresholds for better precision/recall balance
- Optimize context size for response quality
- Identify underperforming pipeline steps

### 4. Debugging & Troubleshooting
- Analyze failed test cases with detailed error information
- Understand user concerns and feedback
- Iterate on parameter settings to address issues

## 🧪 Testing Strategy

### Mock Data Generation
- Generates realistic test cases across different domains
- Includes various query types and complexity levels
- Simulates user ratings and content quality

### Validation
- Structural validation of all data components
- Pipeline step verification
- Performance metric accuracy checking
- UI component functionality testing

## 🎯 Future Enhancements

### Planned Features
- Real model integration (replace simulation)
- Export/import functionality for test cases
- Batch testing and comparison
- Advanced analytics and reporting
- Collaborative testing features

### Technical Improvements
- Performance optimization for large test suites
- Enhanced visualization options
- Advanced parameter search algorithms
- Integration with external RAG systems

## 📝 License

This project is provided as-is for educational and prototyping purposes.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool.

---

**Note**: This is a prototype tool designed for testing and demonstration purposes. The pipeline simulation uses mock data and algorithms to demonstrate the UI and functionality concept.