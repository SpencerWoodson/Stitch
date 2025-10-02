from llama_cpp import Llama
import os

class LLMManager:
    def __init__(self, config):
        self.config = config
        self.model = None

    def load_model(self, mode=None):
        if mode is None:
            mode = self.config.get("force_mode", "fast")
        model_path = os.path.expanduser(self.config["models"][mode])

        print(f"Loading {mode} model from {model_path}...")

        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=7 if self.config["gpu_offloading"] else 0,
            verbose=self.config["verbose_model_loading"],
            n_ctx=self.config.get("context_window", 4096),  # add this
        )


    def ask(self, prompt, stream=False):
        if self.model is None:
            self.load_model()
        if stream:
            return self._stream_answer(prompt)
        else:
            output = self.model(prompt, max_tokens=self._get_max_tokens())
            return output["choices"][0]["text"]

    def _stream_answer(self, prompt):
        """Stream tokens live, then clear raw stream and re-render Markdown nicely."""
        from rich.console import Console
        console = Console()

        answer = ""
        live_stream = self.config.get("live_stream_output", True)

        if live_stream:
            print("[Streaming live output...]\n")

        for chunk in self.model(prompt, stream=True, max_tokens=self._get_max_tokens()):
            choice = chunk["choices"][0]
            token = choice.get("text", "")

            # Only handle non-empty tokens
            if token.strip():
                answer += token
                if live_stream:
                    print(token, end="", flush=True)

            # Check for finish
            if choice.get("finish_reason") is not None:
                break

        if live_stream:
            # Clear the raw streamed text
            print("\033[F\033[K" * (answer.count("\n") + 2), end="")  # move up & clear lines
            print("[Markdown output rendered below]\n")

        return answer



    def _get_max_tokens(self):
        mode = self.config.get("force_mode", "fast")
        return self.config["max_tokens_fast"] if mode == "fast" else self.config["max_tokens_deep"]
