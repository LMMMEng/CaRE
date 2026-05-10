import sys
import time
import logging
import copy
import torch
import shutil
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args):
    torch.cuda.empty_cache()
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    cur_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = f'logs/{args["model_name"]}/{args["prefix"]}/{cur_time}/{args["dataset"]}/{init_cls}/{args["increment"]}/'
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    
    shutil.copy(args["config"], logs_name)

    logfilename = f'{logs_name}/summary'

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    # print_forget = args.get('print_forget', True)
    args["print_forget"] = args.get("print_forget", True)
    model = factory.get_model(args["model_name"], args)

    logging.info("Train Augmentation {}".format(data_manager._train_trsf))
    logging.info("Test Augmentation {}".format(data_manager._test_trsf))

    topk = min(5, args['init_cls'])
    topk = f'top{topk}'

    cnn_curve, nme_curve = {"top1": [], topk: []}, {"top1": [], topk: []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager.nb_tasks):

        # print('\n')
        print(f'exp: {args["config"]}')
        print(f'log: {logfilename}.log \n')

        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        save_path = f'{logs_name}/latest_model.pth'
        print(f'Saving latest model to {save_path}')
        torch.save(model._network.state_dict(), save_path)

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve[topk].append(cnn_accy[topk])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve[topk].append(nme_accy[topk])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN {} curve: {}".format(topk, cnn_curve[topk]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME {} curve: {}\n".format(topk, nme_curve[topk]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve[topk].append(cnn_accy[topk])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN {} curve: {}\n".format(topk, cnn_curve[topk]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    if args["print_forget"]:
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (CNN):')
            print(np_acctable)
            logging.info('Forgetting (CNN): {}'.format(forgetting))
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NME):')
            print(np_acctable)
            logging.info('Forgetting (NME): {}'.format(forgetting))



def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))