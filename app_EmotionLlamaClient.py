"""Lightweight Gradio frontend backed by the reusable inference runtime."""

import argparse
from functools import partial

import gradio as gr

from minigpt4.inference import EmotionLLaMARuntime


_default_runtime = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Emotion-LLaMA Gradio API")
    parser.add_argument(
        "--cfg-path",
        default="eval_configs/demo.yaml",
        help="path to the inference configuration file",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="configuration overrides in xxx=yyy format",
    )
    parser.add_argument(
        "--device",
        help=(
            "Torch device, for example cuda or cuda:1; cpu requires "
            "model.low_resource=false"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="server bind address")
    parser.add_argument("--port", type=int, default=7889, help="server port")
    parser.add_argument("--share", action="store_true", help="create a Gradio share URL")
    return parser.parse_args(argv)


def get_default_runtime():
    """Return an import-safe default runtime for legacy Python integrations."""

    global _default_runtime
    if _default_runtime is None:
        _default_runtime = EmotionLLaMARuntime()
    return _default_runtime


def process_video_question(video_path, question, *, runtime=None):
    """Analyze one server-local video path for the Gradio interface."""

    runtime = runtime or get_default_runtime()
    return runtime.analyze(video_path, question)


def build_interface(runtime):
    predict = partial(process_video_question, runtime=runtime)
    return gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(
                label="Video path",
                placeholder="Enter a server-local video path, such as /path/to/video.mp4",
            ),
            gr.Textbox(
                label="Question",
                placeholder="For example: What emotion does the video convey?",
            ),
        ],
        outputs=gr.Textbox(label="Model answer"),
        title="Emotion-LLaMA API",
        description="Enter a video path and prompt for Emotion-LLaMA inference.",
    )


def main(argv=None):
    args = parse_args(argv)
    runtime = EmotionLLaMARuntime(
        args.cfg_path,
        options=args.options,
        device=args.device,
    )
    interface = build_interface(runtime)
    interface.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
