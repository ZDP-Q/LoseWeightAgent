from jinja2 import Environment, FileSystemLoader, select_autoescape
import os


class PromptManager:
    def __init__(self, template_dir: str = None):
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), "prompts")

        self.env = Environment(
            loader=FileSystemLoader(template_dir), autoescape=select_autoescape()
        )

    def render(self, template_name: str, **kwargs) -> str:
        template = self.env.get_template(template_name)
        return template.render(**kwargs)
