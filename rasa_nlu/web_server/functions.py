def parse(server, request_params):
    data = server.data_router.extract(request_params)
    response_data = server.data_router.parse(data)
    return response_data
