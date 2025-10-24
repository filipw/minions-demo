# MINIONS Protocol Demo

This project is a simple Python demonstration of the **MINIONS** protocol, a method for cost-efficient collaboration between on-device and cloud-based Large Language Models (LLMs). The script uses Apple's MLX framework to run a local "minion" model and the Azure OpenAI API for the powerful remote "manager" model.

## Concept

This demo implements the "divide and conquer" strategy described in the research paper "[Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models](https://arxiv.org/abs/2502.15964v1)" by Narayan et al.

The core idea is a cost-efficient collaboration between a small, on-device language model (the "minion") and a powerful, cloud-hosted LM (the "manager"). The goal is to solve tasks involving long documents while minimizing expensive cloud API calls. The manager model breaks down a complex query into simple, parallelizable subtasks ("jobs") that are executed by the minion across small chunks of the local document.

## Requirements

* **Hardware**: **A Mac with Apple Silicon is required** to run this demo, as it uses Apple's MLX framework for local model inference.
* **Python**: 3.9+
* **Azure OpenAI Access**: An active Azure OpenAI subscription with a smart deployed model (e.g., GPT-5).

The model running locally should be a small, efficient LLM. In this demo, we use the `Phi-4-mini-instruct-8bit` model from Hugging Face, which is quantized to 8-bit for efficiency and optimized for MLX.

## Setup & Configuration

1.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure your environment variables:**
    Create a file named `.env` in the root of the project and add your Azure OpenAI credentials:

    ```env
    AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com"
    AZURE_OPENAI_API_KEY="your-secret-api-key"
    AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"
    ```

## Usage

Once the setup is complete, simply run the demo script from your terminal: `python demo.py`

## How It Works

The demo executes the MINIONS protocol in three main steps:

1.  **Step 1: Job Preparation:** The remote "manager" Azure OpenAI model creates jobs by generating code that is executed locally.
2.  **Step 2: Job Execution:** The local "minion" model (running on-device via MLX) runs the jobs on document chunks and filters the results.
3.  **Step 3: Aggregation & Synthesis:** The remote "manager Azure OpenAI model receives the filtered results and provides the final answer.

## Acknowledgements

This project is a practical implementation of the concepts described in the research paper:

> Narayan, A., Biderman, D., Eyuboglu, S., May, A., Linderman, S., Zou, J., & Ré, C. (2025). *Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models*. arXiv preprint arXiv:2502.15964.