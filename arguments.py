import argparse

def add_LM_forward_api_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('fwd_api', 'forward api arguments')
    group.add_argument("--task_name", default='easy', type=str)
    group.add_argument("--intrinsic_dim", default=500, type=int)
    group.add_argument("--budget", default=8000, type=int)
    group.add_argument("--device", default='cuda:0', type=str)
    group.add_argument("--alg", default='CMA', type=str)
    group.add_argument("--random_proj", default='normal', type=str)  # normal or he
    group.add_argument("--seed", default=42, type=int)
    group.add_argument(
        "--inference_framework",
        default='pt',
        type=str,
        help='''Which inference framework to use. 
             Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
    )
    group.add_argument(
        "--onnx_model_path",
        default=None,
        type=str,
        help='Path to your onnx model.'
    )
    return parser

def add_optim_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('optim', 'optimization algorithm arguments')
    group.add_argument("--popsize", default=20, type=int)  # cma
    group.add_argument("--bound", default=0, type=int)  # cma, bo
    group.add_argument("--sigma", default=1, type=float)  # cma, bo
    group.add_argument("--kappa", default=2., type=float)  # bo
    group.add_argument("--acq", default='ucb', type=str)  # bo
    group.add_argument("--init_points", default=50, type=int)  # bo
    return parser

def get_arguments():
    parser = argparse.ArgumentParser()
    parser = add_LM_forward_api_args(parser)
    parser = add_optim_args(parser)
    args = parser.parse_args()
    return args
