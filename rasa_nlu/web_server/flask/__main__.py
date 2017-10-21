import argparse
import logging
import os

from flask import Flask

from rasa_nlu.web_server.flask.blueprint.data_router import DataRouter
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.web_server.flask.blueprint.main import api_app

app = Flask(__name__)


logger = logging.getLogger(__file__)


def create_argparser():
    parser = argparse.ArgumentParser(description='parse incoming text')
    parser.add_argument('-c', '--config',
                        help="config file, all the command line options can "
                             "also be passed via a (json-formatted) config "
                             "file. NB command line args take precedence")
    parser.add_argument('-e', '--emulate',
                        choices=['wit', 'luis', 'api'],
                        help='which service to emulate (default: None i.e. use '
                             'simple built in format)')
    parser.add_argument('-l', '--language',
                        choices=['de', 'en'],
                        help="model and data language")
    parser.add_argument('-m', '--mitie_file',
                        help='file with mitie total_word_feature_extractor')
    parser.add_argument('-p', '--path',
                        help="path where project files will be saved")
    parser.add_argument('--pipeline',
                        help="The pipeline to use. Either a pipeline template "
                             "name or a list of components separated by comma")
    parser.add_argument('-P', '--port',
                        type=int,
                        help='port on which to run server')
    parser.add_argument('-t', '--token',
                        help="auth token. If set, reject requests which don't "
                             "provide this token as a query parameter")
    parser.add_argument('-w', '--write',
                        help='file where logs will be saved')

    return parser


def builder_server(config, component_builder=None, testing=False):
    class Server(object):
        pass

    server = Server()

    logging.basicConfig(filename=config['log_file'],
                        level=config['log_level'])
    logging.captureWarnings(True)
    logger.debug("Configuration: " + config.view())

    logger.debug("Creating a new data router")
    server.config = config
    server.data_router = _create_data_router(config, component_builder)
    server._testing = testing
    # TODO: config['num_threads'] is not used
    return server


def _create_data_router(config, component_builder):
    return DataRouter(config, component_builder)

# Running as standalone python application
arg_parser = create_argparser()
cmdline_args = {key: val
                for key, val in list(vars(arg_parser.parse_args()).items())
                if val is not None}
rasa_nlu_config = RasaNLUConfig(
        cmdline_args.get("config"), os.environ, cmdline_args)
rasa_nlu_server_config = builder_server(rasa_nlu_config)

app.config['rasa_nlu_server_config'] = rasa_nlu_server_config

app.register_blueprint(api_app, url_prefix='')

logger.info('Started http server on port %s' % rasa_nlu_config['port'])
app.run('0.0.0.0', rasa_nlu_config['port'])