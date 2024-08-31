# Legal Contract Analysis Assistant

## Overview

This project implements a Legal Contract Analysis Assistant using Chainlit, LangChain, and GPT-4. The assistant specializes in analyzing vendor contracts, comparing them with existing ones, and suggesting improvements based on vendor performance history.

## Features

1. PDF Contract Upload and Analysis
2. Chat-based Interaction
3. Vendor Performance Analysis
4. Contract Comparison
5. Improvement Suggestions
6. Authentication System

## Dependencies

- Python 3.11+
- Chainlit
- LangChain
- OpenAI GPT-4
- PyPDF2 (for PDF processing)
- LiteralAI (for analytics)
- Guardrails (for guardrails)

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   - see `.env.example`

## Usage

Run the application:

- `chainlit run main.py`
