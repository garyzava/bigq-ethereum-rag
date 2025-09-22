# ui.py
import gradio as gr
from google.cloud import bigquery
from bigq_ai_app.core.config import BaseConfig
from bigq_ai_app.core.orchestrator import run_orchestrator

def create_ui(phase: str = "auto"):
    """Creates a Gradio UI based on the phase: architect, semantic, multimodal, auto."""
    project_id = BaseConfig.PROJECT_ID
    location = BaseConfig.LOCATION
    client = bigquery.Client(project=project_id, location=location)

    # Just decide titles/descriptions here
    if phase == "architect":
        title = "BigQuery Architect Chat with Ethereum"
        description = "Simple query generation"
    elif phase == "semantic":
        title = "BigQuery Semantic Chat with Ethereum"
        description = "With retrieval from ingested docs"
    elif phase == "multimodal":
        title = "BigQuery Multimodal Chat with Ethereum"
        description = "With hybrid retrieval and file uploads"
    elif phase == "auto":
        title = "BigQuery AI Chat with Ethereum"
        description = "Intelligent query processing with intent classification"
    else:
        raise ValueError("Invalid phase")

    with gr.Blocks() as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1):
                # Build inputs INSIDE the layout
                if phase == "architect":
                    q_in = gr.Textbox(label="Your Question")
                    inputs = [q_in]
                elif phase == "semantic":
                    q_in = gr.Textbox(label="Your Question (with intelligent processing)")
                    inputs = [q_in]
                elif phase in ["multimodal", "auto"]:
                    q_in = gr.Textbox(label="Your Question (multimodal with hybrid retrieval)")
                    files_in = gr.File(label="Upload Files", file_count="multiple")
                    inputs = [q_in, files_in]

                submit_button = gr.Button("Submit")

            with gr.Column(scale=2):
                sql_output = gr.Textbox(label="Generated SQL", interactive=False, lines=8)
                results_output = gr.Dataframe(label="Results")
                notes_output = gr.Textbox(label="Notes/Logs", interactive=False, lines=6)

        def handle_submit(*args):
            if phase in ["multimodal","auto"]:
                q, files = args
                file_paths = [f.name for f in files] if files else []
                sql, results_df, notes = run_orchestrator(
                    q, client, project_id, location, phase, uploaded_files=file_paths
                )
            else:
                (q,) = args
                sql, results_df, notes = run_orchestrator(
                    q, client, project_id, location, phase
                )
            return sql, results_df, notes

        # Click and Enter-to-submit
        submit_button.click(handle_submit, inputs=inputs, outputs=[sql_output, results_output, notes_output])
        q_in.submit(handle_submit, inputs=inputs, outputs=[sql_output, results_output, notes_output])

    return demo