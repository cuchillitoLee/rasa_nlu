from .project_loader import ProjectLoader


class InplaceParsing(object):
    def __init__(self, config, query_logger=None):
        self.query_logger = query_logger
        self.config = config

    def parse(self, text, project, model, time=None):
        project_loader = ProjectLoader(self.config)
        project = project_loader.load_project(project)

        response, used_model = project.parse(text, time, model)

        if self.query_logger:
            self.query_logger.info('', user_input=response, project=project, model=used_model)
        return response
