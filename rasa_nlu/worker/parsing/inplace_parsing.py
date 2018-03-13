from .project_loader import ProjectLoader
from rasa_nlu.components import ComponentBuilderWithDebugHelper


class InplaceParsing(object):
    def __init__(self, config, query_logger=None):
        self.query_logger = query_logger
        self.config = config

        self.project_loader = ProjectLoader(self.config, ComponentBuilderWithDebugHelper(use_cache=True))

    def parse(self, text, project, model, time=None):
        project = self.project_loader.load_project(project)

        response, used_model = project.parse(text, time, model)

        if self.query_logger:
            self.query_logger.info('', user_input=response, project=project, model=used_model)
        return response
