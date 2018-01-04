from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import traceback
from functools import wraps

import simplejson
from flask import Response, Blueprint, request, current_app

from rasa_nlu.web_server.flask.blueprint.data_router import InvalidProjectError, AlreadyTrainingError
from rasa_nlu.train import TrainingException
from rasa_nlu.version import __version__
from rasa_nlu.web_server import functions

logger = logging.getLogger(__name__)

api_app = Blueprint('api_app', __name__, template_folder='templates')


def check_cors(f):
    """Wraps a request handler with CORS headers checking."""

    @wraps(f)
    def decorated(*args, **kwargs):
        server = current_app.config['rasa_nlu_server_config']

        origin = request.headers.get('Origin')

        if origin:
            resp = Response("forbidden")
            resp.headers['Access-Control-Allow-Origin'] = '*'

            if '*' in server.config['cors_origins']:
                resp.headers['Access-Control-Allow-Origin'] = '*'
            elif origin in server.config['cors_origins']:
                resp.headers['Access-Control-Allow-Origin'] = origin
            else:
                resp.status_code = 403
                return resp

        return f(*args, **kwargs)

    return decorated


def requires_auth(f):
    """Wraps a request handler with token authentication."""

    @wraps(f)
    def decorated(*args, **kwargs):
        token = str(request.args.get('token', [''])[0])

        server = current_app.config['rasa_nlu_server_config']

        if server.data_router.token is None or token == server.data_router.token:
            return f(*args, **kwargs)

        resp = Response("unauthorized")
        resp.status_code = 401

        return resp

    return decorated


@api_app.route("/", methods=['GET'])
@check_cors
def hello():
    """Main Rasa route to check if the server is online"""
    return "hello from Rasa NLU: " + __version__


@api_app.route("/parse", methods=['GET', 'POST'])
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
        server = current_app.config['rasa_nlu_server_config']

        resp = Response()
        resp.headers['Access-Control-Allow-Origin'] = '*'

        data = server.data_router.extract(request_params)

        try:
            resp.status_code = 200
            response_data = server.data_router.parse(data)
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


@api_app.route("/version", methods=['GET'])
@requires_auth
@check_cors
def version():
    """Returns the Rasa server's version"""
    resp = Response()
    resp.headers['Content-Type'] = 'application/json'
    resp.response = simplejson.dumps({'version': __version__})
    return resp


@api_app.route("/config", methods=['GET'])
@requires_auth
@check_cors
def rasaconfig():
    """Returns the in-memory configuration of the Rasa server"""
    server = current_app.config['rasa_nlu_server_config']

    resp = Response()
    resp.headers['Content-Type'] = 'application/json'
    resp.response = simplejson.dumps(server.config.as_dict())
    return resp


@api_app.route("/status", methods=['GET'])
@requires_auth
@check_cors
def status():
    server = current_app.config['rasa_nlu_server_config']

    resp = Response()
    resp.headers['Content-Type'] = 'application/json'
    resp.response = simplejson.dumps(server.data_router.get_status())
    return resp


@api_app.route("/train", methods=['POST'])
@requires_auth
@check_cors
def train():
    data_string = request.data.decode('utf-8', 'strict')
    kwargs = {key: value for key, value in request.args.items()}

    server = current_app.config['rasa_nlu_server_config']

    resp = Response()
    resp.headers['Content-Type'] = 'application/json'

    try:
        resp.status_code = 200
        response = server.data_router.start_train_process(
            data_string, kwargs)
        resp.data = simplejson.dumps(
            {'info': 'new model trained: {}'.format(response)})
        return resp
    except AlreadyTrainingError as e:
        resp.status_code = 403
        resp.data = simplejson.dumps({"error": "{}".format(e)})
        return resp
    except InvalidProjectError as e:
        resp.status_code = 404
        resp.data = simplejson.dumps({"error": "{}".format(e)})
        return resp
    except TrainingException as e:
        logger.exception(e.raw_exception)
        resp.status_code = 500
        resp.data = simplejson.dumps(
            {"error": "{}".format(e)})
        return resp
