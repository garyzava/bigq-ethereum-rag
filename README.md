# BigQuery Chat with Ethereum

## Prerequisites

This guide uses the **Application Default Credentials (ADC)** method.

1. **Set up a Google Cloud project**  
   - [Create a Google Cloud project](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart?usertype=adc#setup-gcp)  
   - [Enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com)

2. **Install the gcloud CLI**  
   - macOS: `brew install --cask google-cloud-sdk`  
   - Windows: [gcloud CLI installer](https://cloud.google.com/sdk/docs/install)

## Requirements

1. **Environment variables**  
   Create a `.env` file with your project and location settings.  
   - Recommended: use `US` as the location since public datasets are in the US region (this avoids cross-regional issues).  
   - Use `.env.example` as a template.  

2. **Authenticate with Google Cloud**  
   Run the following commands in your terminal:

   ```bash
   # Load environment variables
   set -a; source .env; set +a;

   # Authenticate
   gcloud auth application-default login

   # Set project
   gcloud config set project "$GOOGLE_CLOUD_PROJECT"
   ```

3. **Create a BigQuery connection**
    ```bash
    bq mk --connection \
    --project_id="$GOOGLE_CLOUD_PROJECT" \
    --location="$GOOGLE_CLOUD_LOCATION" \
    --connection_type=CLOUD_RESOURCE \
    bq-llm-connection || true

    echo '---'

    bq show --connection \
    --project_id="$GOOGLE_CLOUD_PROJECT" \
    --location="$GOOGLE_CLOUD_LOCATION" \
    bq-llm-connection || true
    ```

   More on BigQuery connections [here](https://cloud.google.com/bigquery/docs/working-with-connections#bq)

## Troubleshooting

If you encounter the following error while running BigQuery ML models:

```
The xyz@gcp-sa-bigquery-condel.iam.gserviceaccount.com does not have the permission to access or use the endpoint. Please grant the Vertex AI user role to the xyz@gcp-sa-bigquery-condel.iam.gserviceaccount.com following https://cloud.google.com/bigquery/docs/generate-text-tutorial#grant-permissions. If issue persists, contact bqml-feedback@google.com for help.
```

Grant the **Vertex AI User** role to the service account:

* Follow [this guide](https://cloud.google.com/bigquery/docs/generate-text-tutorial#grant-permissions).

## Structure
```
.
├── bigq_ai_app/
│   ├── core/
│   ├── utils/
│   └── ui.py
├── docs/
├── notebooks/
├── tests/
├── .env.example
├── .gitignore
├── app.py
├── pyproject.toml
└── README.md

```

## How to run it

1. Create a virtual environment:
   ```bash
   uv venv
   ```
2. Activate the environment:
   ```bash
   source .venv/bin/activate
   # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies from your ```pyproject.toml```:
   ```bash
   uv pip install -e .
   ```
   (The -e . installs your project in "editable" mode, which is great for development).

4. Run the Gradio app:
   ```bash
   python app.py
   ```

   * Open http://127.0.0.1:7865
   * To expose a public link quickly: set share=True in launch()
   * To bind for containers/VMs: use server_name="0.0.0.0   