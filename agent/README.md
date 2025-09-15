# Multi-Agent Travel Planning System

An autonomous multi-agent system for travel planning that leverages LLMs (Gemini/OpenAI) to create comprehensive travel itineraries based on Points of Interest (POI) and video analysis.

## Prerequisites

### System Requirements

- **Python**: 3.12

### Required Software

**Chromium/Chrome Browser**
   - Required for web scraping and Google Maps integration
   - Install via:
     ```bash
     # macOS
     brew install --cask chromium

     # Ubuntu/Debian
     sudo apt-get install chromium-browser

     # Windows
     # Download from https://www.chromium.org/
     ```

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:nlp-waseda/VIR-Bench.git
cd VIR-BENCH/agent
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt
```

### 3. Configure API Keys

Create a `.env` file in the agent directory:

```bash
cp .env.sample .env
```

Edit `.env` and add your API keys:

```env
# Required for Gemini models
GOOGLE_API_KEY=your-google-api-key-here

# Required for OpenAI models
OPENAI_API_KEY=your-openai-api-key-here

# Required for Google Maps features
GOOGLE_MAPS_API_KEY=your-google-maps-api-key-here
```

#### Getting API Keys:
- **Google API Key**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **OpenAI API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Google Maps API Key**: [Google Cloud Console](https://console.cloud.google.com/apis/credentials)

## Quick Start

### Basic Usage

1. **Using POI List**:
```bash
python src/main.py --poi-list "Tokyo Tower" "Senso-ji Temple" "Shibuya Crossing"
```

2. **Using Video Analysis** (Gemini only):
```bash
python src/main.py --video-path travel_vlog.mp4 --model-type gemini
```

3. **Using Video ID with POI extraction**:
```bash
python src/main.py --video-id UU8YnZ7iBOw --use-poi
```

### Advanced Options

```bash
# Specify model and parameters
python src/main.py \
  --model-type gemini \
  --model-name gemini-2.5-pro \
  --temperature 0.7 \
  --people-count 2 \
  --days 3 \
  --budget-usd 2000 \
  --location "Tokyo, Japan" \
  --poi-list "Tokyo Tower" "Asakusa"
```

### Available Models

View supported models:
```bash
python src/main.py --list-models
```

Supported providers:
- **Gemini**: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash
- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5-turbo

## Features

### 1. Multi-Agent Architecture
- **Google Maps Agent**: Collects detailed POI information
- **Plan Agent**: Creates day-by-day itineraries
- **Route Search Agent**: Finds optimal routes between locations
- **Accommodation Agent**: Searches for suitable hotels
- **Summary Agent**: Generates comprehensive travel plans

### 2. Input Methods
- **POI List**: Direct specification of places to visit
- **Video Analysis**: Extract POIs from travel videos (Gemini only)
- **Graph Data**: Use pre-processed POI graphs from video_id

### 3. Autonomous Orchestration
- Dynamic agent coordination
- Intelligent decision making
- Budget-aware planning
- Context-aware recommendations

## Output Structure

```
results/
└── experiment_YYYYMMDD_HHMMSS/
    ├── experiment_config.json      # Input parameters
    ├── model_config.json           # Model configuration
    ├── raw_data/                  # Agent outputs
    │   ├── *_google_map_agent.json
    │   ├── *_plan_agent.json
    │   └── ...
    ├── tool_logs/                 # Tool execution logs
    ├── orchestration_log.json     # Decision history
    ├── final_plan.json            # Complete travel plan
    └── token_usage.json           # Token usage statistics
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all required API keys are set in `.env`
   - Check API key permissions and quotas

2. **Video Processing Fails**
   - Verify video format is MP4
   - Check file size (recommended < 100MB)
   - Ensure Gemini model is selected

3. **Chromium Not Found**
   - Install Chromium/Chrome browser
   - Set CHROME_EXECUTABLE_PATH if needed

4. **Module Import Errors**
   ```bash
   pip install --upgrade -r requirement.txt
   ```

### Debug Mode

Enable verbose logging:
```bash
export DEBUG=1
python src/main.py --poi-list "Tokyo Tower"
```

## Project Structure

```
agent/
├── src/
│   ├── main.py              # Entry point
│   ├── agents.py            # Agent implementations
│   ├── tools.py             # Tool definitions
│   ├── prompt.py            # LLM prompts
│   ├── model_config.py      # Model configuration
│   ├── token_counter.py     # Token usage tracking
│   ├── output_manager.py    # Output handling
│   └── video_analyzer.py    # Video processing
├── data/                    # Input data
│   ├── graphs_v2/          # POI graphs
│   └── videos/             # Video files
├── results/                # Experiment outputs
├── experiment_logs/        # Execution logs
├── requirement.txt         # Python dependencies
├── .env.sample            # Environment template
└── run_experiments.zsh    # Batch experiment script
```