import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default='easy', type=str)
    parser.add_argument("--intrinsic_dim", default=500, type=int)
    parser.add_argument("--budget", default=8000, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--alg", default='CMA', type=str)
    parser.add_argument("--random_proj", default='normal', type=str)  # normal or he
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--popsize", default=20, type=int)  # cma
    parser.add_argument("--bound", default=5, type=int)  # cma, bo
    parser.add_argument("--sigma", default=1, type=float)  # cma, bo
    parser.add_argument("--kappa", default=2., type=float)  # bo
    parser.add_argument("--acq", default='ucb', type=str)  # bo
    parser.add_argument("--init_points", default=50, type=int)  # bo
    parser.add_argument(
        "--inference_framework",
        default='pt',
        type=str,
        help='''Which inference framework to use. 
             Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
    )
    parser.add_argument(
        "--onnx_model_path",
        default=None,
        type=str,
        help='Path to your onnx model.'
    )
    args = parser.parse_args()
    return args
