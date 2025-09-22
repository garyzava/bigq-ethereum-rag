# main.py

#from bigq_ai_app.core.ai_architect_ui import create_ui
#from bigq_ai_app.core.ai_semantic_ui import create_semantic_ui
from bigq_ai_app.ui import create_ui
from bigq_ai_app.core.config import BaseConfig

if __name__ == "__main__":
    demo = create_ui("auto")  # options: "architect", "semantic", "multimodal", "auto"
    demo.launch(server_name="127.0.0.1", server_port=BaseConfig.GRADIO_PORT, share=False)
