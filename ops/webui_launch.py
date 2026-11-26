"""Build and launch the Gradio UI, blocking. Started detached by webui_cpu_serve.py."""
import sys

CONTENT = "/content"
sys.path.insert(0, CONTENT)

if __name__ == "__main__":
    import anima_gradio_ui as ui

    demo = ui.build()
    print("launching", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                quiet=False, show_error=True)
