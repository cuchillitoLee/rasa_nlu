from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import traceback

from builtins import object

from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import InvalidProjectError
from rasa_nlu.project import Project

logger = logging.getLogger(__name__)


class ProjectLoader(object):
    DEFAULT_PROJECT_NAME = "default"

    def __init__(self, config):
        self.config = config
        self.component_builder = ComponentBuilder(use_cache=True)
        self.project_store = self._create_project_store()

    def _create_project_store(self):
        projects = []

        if os.path.isdir(self.config['path']):
            # TODO: start a pull request
            projects = [i for i in os.listdir(self.config['path']) if os.path.isdir(os.path.join(self.config['path'], i))]

        cloud_provided_projects = self._list_projects_in_cloud()

        projects.extend(cloud_provided_projects)

        project_store = {}

        for project in projects:
            project_store[project] = Project(self.config, self.component_builder, project)

        if not project_store:
            project_store[RasaNLUConfig.DEFAULT_PROJECT_NAME] = Project(self.config)
        return project_store

    def _list_projects_in_cloud(self):
        try:
            from rasa_nlu.persistor import get_persistor
            p = get_persistor(self.config)
            if p is not None:
                return p.list_projects()
            else:
                return []
        except Exception as e:
            logger.warning("Failed to list projects. {}".format(traceback.format_exc()))
            return []

    def load_project(self, project=None):
        project = project or RasaNLUConfig.DEFAULT_PROJECT_NAME

        if project not in self.project_store:
            projects = self._list_projects(self.config['path'])
            logger.info("projects: {}".format(projects))

            cloud_provided_projects = self._list_projects_in_cloud()
            logger.info("cloud_provided_projects: {}".format(cloud_provided_projects))

            projects.extend(cloud_provided_projects)

            if project not in projects:
                raise InvalidProjectError("No project found with name '{}'.".format(project))
            else:
                try:
                    self.project_store[project] = Project(self.config, self.component_builder, project)
                except Exception as e:
                    raise InvalidProjectError("Unable to load project '{}'. Error: {}".format(project, e))

        return self.project_store[project]


