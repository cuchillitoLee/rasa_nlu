from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import logging
import os
import traceback
from functools import wraps

import simplejson

from flask import Flask, Response
from flask import request

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.data_router import DataRouter, InvalidProjectError
from rasa_nlu.version import __version__

logger = logging.getLogger(__name__)


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


app = Flask(__name__)


class Server(object):
    pass


server = Server()


def check_cors(f):
    """Wraps a request handler with CORS headers checking."""

    @wraps(f)
    def decorated(*args, **kwargs):
        origin = request.headers.get('Origin')

        if origin:
            resp = Response("forbidden")
            resp.headers['Access-Control-Allow-Origin'] = '*'

            if '*' in server.config['cors_origins']:
                request.headers['Access-Control-Allow-Origin'] = '*'
            elif origin in server.config['cors_origins']:
                request.headers['Access-Control-Allow-Origin'] = origin
            else:
                request.status_code = 403
                return resp

        return f(*args, **kwargs)

    return decorated


def requires_auth(f):
    """Wraps a request handler with token authentication."""

    @wraps(f)
    def decorated(*args, **kwargs):
        token = str(request.args.get('token', [''])[0])

        if server.data_router.token is None or token == server.data_router.token:
            return f(*args, **kwargs)
        request.status_code = 401
        return 'unauthorized'

    return decorated


def builder_server(config, component_builder=None, testing=False):
    logging.basicConfig(filename=config['log_file'],
                        level=config['log_level'])
    logging.captureWarnings(True)
    logger.debug("Configuration: " + config.view())

    logger.debug("Creating a new data router")
    server.config = config
    server.data_router = _create_data_router(config, component_builder)
    server._testing = testing
    # TODO: config['num_threads'] is not used


def _create_data_router(config, component_builder):
    return DataRouter(config, component_builder)


@app.route("/", methods=['GET'])
@check_cors
def hello():
    """Main Rasa route to check if the server is online"""
    return "hello from Rasa NLU: " + __version__


@app.route("/parse", methods=['GET', 'POST'])
@requires_auth
@check_cors
def parse_get():
    request.headers.get('Content-Type', 'application/json')
    if request.method == 'GET':
        request_params = {key: value for key, value in request.args.items()}
    else:
        request_params = simplejson.loads(
                request.data.decode('utf-8', 'strict'))

    if 'query' in request_params:
        request_params['q'] = request_params.pop('query')

    if 'q' not in request_params:
        request.status_code = 404
        dumped = simplejson.dumps({
            "error": "Invalid parse parameter specified"})
        return dumped
    else:
        data = server.data_router.extract(request_params)

        resp = Response()
        resp.headers['Access-Control-Allow-Origin'] = '*'

        try:
            resp.status_code = 200
            response_data = server.data_router.parse(data) if server._testing else server.data_router.parse(data)
            resp.response = simplejson.dumps(response_data)
            return resp
        except InvalidProjectError as e:
            resp.status_code = 404
            resp.response = simplejson.dumps({"error": "{}".format(e)})
            return resp
        except Exception as e:
            traceback.print_exc()
            print(e)
            resp.status_code = 500
            resp.response = simplejson.dumps({"error": "{}".format(e)})
            return resp


@app.route("/version", methods=['GET'])
@requires_auth
@check_cors
def version():
    """Returns the Rasa server's version"""
    resp = Response()
    resp.headers['Content-Type'] = 'application/json'
    resp.response = simplejson.dumps({'version': __version__})
    return resp


@app.route("/config", methods=['GET'])
@requires_auth
@check_cors
def rasaconfig():
    """Returns the in-memory configuration of the Rasa server"""
    resp = Response()
    resp.headers['Content-Type'] = 'application/json'
    resp.response = simplejson.dumps(server.config.as_dict())
    return resp


@app.route("/status", methods=['GET'])
@requires_auth
@check_cors
def status():
    resp = Response()
    resp.headers['Content-Type'] = 'application/json'
    resp.response = simplejson.dumps(server.data_router.get_status())
    return resp


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argparser()
    cmdline_args = {key: val
                    for key, val in list(vars(arg_parser.parse_args()).items())
                    if val is not None}
    rasa_nlu_config = RasaNLUConfig(
            cmdline_args.get("config"), os.environ, cmdline_args)
    builder_server(rasa_nlu_config)
    logger.info('Started http server on port %s' % rasa_nlu_config['port'])
    app.run('0.0.0.0', rasa_nlu_config['port'])
