import onnxruntime as ort

VALID_PROVIDERS = set(['cpu', 'cuda', 'openvino'])


def create_session(model_path, provider, num_threads=1):
    assert provider in VALID_PROVIDERS

    providers = None
    provider_options = None

    if provider == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif provider == 'openvino':
        from openvino import utils
        utils._add_openvino_libs_to_search_path()
        providers = ['OpenVINOExecutionProvider']
    else:
        providers = [ 'CPUExecutionProvider' ]

    sess_options = ort.SessionOptions()
    # sess_options.intra_op_num_threads = num_threads

    session = ort.InferenceSession(
        model_path, sess_options, providers=providers
    )
    return session
