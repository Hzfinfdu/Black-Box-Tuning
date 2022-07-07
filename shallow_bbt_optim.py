import time
import os
from bbt import LMForwardAPI, args, train_data, test_data
import torch
import numpy as np
import cma
from fastNLP import DataSet
from bayes_opt import BayesianOptimization

task_name = args.task_name
n_prompt_tokens = 50  # fixed
intrinsic_dim = args.intrinsic_dim
k_shot = 16  # fixed
batch_size = 32  # fixed
budget = args.budget
alg = args.alg
bound = args.bound
sigma = args.sigma
if args.popsize > 0:
    popsize = args.popsize
else:
    popsize = 4 + 3 * np.log(intrinsic_dim)
device = args.device
kappa = args.kappa  # bo
init_points = args.init_points  # bo
acq = args.acq  # bo
random_proj = args.random_proj
seed = args.seed
cat_or_add = 'add'  # fixed
args.bbt_version = 'bbt'

#  ort may give model forward a ~2x boost, see Readme.md for more details.
inference_framework = args.inference_framework
onnx_model_path = args.onnx_model_path
if inference_framework not in ['pt', 'ort']:
    raise ValueError(f'inference_framework only supports "pt", "ort", got `{inference_framework}` instead.')
if inference_framework == 'ort':
    assert onnx_model_path is not None, 'Path to onnx model is required, got None instead.'
    assert os.path.exists(onnx_model_path), f'In valid onnx model path `{onnx_model_path}`'
# end of model boost

# This param selection is based on previous experiments, not recommended to tune. It has little to do with black box optimization.
loss_type = 'hinge' if task_name == 'hard' else 'ce'

save_path = f"results/{task_name}_results/d_{intrinsic_dim}_alg_{alg}_range_{bound}_loss_{loss_type}_budget_{budget}_seed_{seed}_randomproj_{random_proj}_framework_{inference_framework}"
print(f'Results will be saved in {save_path}')

if os.path.exists(save_path):
    print('Experiment already run.')
    exit()

args.save_path = save_path


model_forward_api = LMForwardAPI(
    intrinsic_dim=intrinsic_dim,
    n_prompt_tokens=n_prompt_tokens,
    task_name=task_name,
    loss_type=loss_type,
    inference_framework=inference_framework,
    onnx_model_path=onnx_model_path,
    save_path=save_path,
    device=device
)

########################################## Notes #######################################################
# Above are hyperparameters. Implementation details are wrapped in LMForwardAPI. Just treat it as a black box :)
# model_forward_api.eval is the fitness function, it takes a numpy.ndarray of size (intrinsic_dim,) as input and gives a real fitness value.
# Below is the baseline method implemented with cma-es (for default arguments see arguments.py). Its performance is reported in the readme :)
# In real-world NLP scenarios, different from previous benchmarks, our BBT method does not mainly focus on fitting but on generalization.
# Therefore, in the black box api, we validate on a validation set (of the same size as the training set) every 100 fitness evaluation and maintain the best param.
# After the training phase we use the best param for testing (on a far larger test set), and get a test score for comparison.
########################################### End ########################################################


# fitness function: model_forward_api.eval(d: np.ndarray of size(intrinsic_dim, )) -> Tuple(fitness_value: int, eval_loss: None | int)
# evaluation is auto-conducted. Element 1 of the return value of the fwd api is eval loss in eval steps and None otherwise
# run model_forward_api.eval(test_data=test_data) for test phase

if alg == 'CMA':
    cma_opts = {
        'seed': seed,
        'popsize': popsize,
        'maxiter': budget // popsize,
        'verbose': -1,
    }
    if bound > 0:
        cma_opts['bounds'] = [-1 * bound, 1 * bound]
    es = cma.CMAEvolutionStrategy(intrinsic_dim * [0], sigma, inopts=cma_opts)
    print('Population Size: {}'.format(es.popsize))

    # opt = cma.CMAOptions()
    start_time = time.time()
    while not es.stop():
        solutions = es.ask()
        fitnesses = [model_forward_api.eval(x)[0] for x in solutions]
        es.tell(solutions, fitnesses)
        # es.logger.add()  # write data to disc to be plotted
        # es.disp()
    end_time = time.time()
    print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
    print('Evaluate on test data...')
    test_acc = model_forward_api.eval(test_data=test_data)
    with open('res.txt', 'a+') as f:
        print(f'{task_name}_{seed}_Test acc: {round(test_acc, 4)}', file=f)
    # fitlog.finish()


elif alg == 'bo':
    bounds_x = [f'x{i}' for i in range(intrinsic_dim * 100, intrinsic_dim * 100 + intrinsic_dim)]
    start_time = time.time()  # 开始训练
    bounds_range = [(-bound, bound)] * intrinsic_dim

    def A_model_forward_api_eval(**kwargs):
        x = np.fromiter(kwargs.values(), dtype=float)
        return -model_forward_api.eval(x)


    pbounds = dict(zip(bounds_x, bounds_range))

    BO_optimizer = BayesianOptimization(
        f=A_model_forward_api_eval,
        pbounds=pbounds,
        verbose=1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=seed,
    )

    BO_optimizer.maximize(init_points=init_points, n_iter=budget - init_points, acq=acq, kappa=kappa)
    best_dict_x = BO_optimizer.max['params']
    best_x = np.fromiter(best_dict_x.values(), dtype=float)
    if best_dict_x != dict(zip(bounds_x, best_x)):
        raise Exception("Invalid level!", [dict(zip(bounds_x, best_x)), best_dict_x])
    # 检查参数是否乱序
    # perf_list = [model_forward_api.eval(x) for x in prompt_list[1:]]
    # model_forward_api.set_best_prompt(prompt_list[np.argmin(perf_list) + 1])  # 取表现最好的x_i
    end_time = time.time()

    print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
    print('Evaluate on test data...')
    test_acc = model_forward_api.eval(test_data=test_data)
    print('Test acc: {}'.format(round(test_acc, 4)))
    with open('result_bo.txt', 'a+') as f:
        print(test_acc, file=f)
