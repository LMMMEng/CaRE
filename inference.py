import sys
import logging
import torch
import numpy as np
import json
import argparse
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from torch.utils.data import DataLoader


def inference(args):
    """
    Load a trained checkpoint and perform inference on test data
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Handle seed if it's a list - fix it in args for DataManager
    seed = args["seed"]
    if isinstance(seed, list):
        seed = seed[0]
        args["seed"] = seed  # Fix it in args so DataManager gets the correct seed
    _set_random(seed)
    
    _set_device(args)
    print_args(args)
    
    # Initialize data manager with corrected seed
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks
    
    logging.info("Test Augmentation {}".format(data_manager._test_trsf))
    
    # Initialize model
    model = factory.get_model(args["model_name"], args)
    
    # Build complete model structure by simulating all tasks
    # This is crucial to ensure the checkpoint can be loaded with strict=True
    logging.info(f"Building model structure for {data_manager.nb_tasks} tasks...")
    
    for task in range(data_manager.nb_tasks):
        model._cur_task = task
        task_size = data_manager.get_task_size(task)
        model._total_classes = model._known_classes + task_size
        
        # Map classes to tasks
        for i in range(model._known_classes, model._total_classes):
            model.cls2task[i] = task
        
        # Update network structure (same as in training)
        num_new_classes = model._total_classes - model._known_classes
        model._network.update_fc(num_new_classes)
        
        # Update adapters for task > 0
        if task > 0:
            model._network.backbone.update_adapters(num_new_classes)
        
        model._known_classes = model._total_classes
        
        logging.info(f"Task {task}: classes {model._total_classes - task_size} -> {model._total_classes}")
    
    # Now load checkpoint with strict=True to ensure complete loading
    checkpoint_path = args.get("checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("Please specify checkpoint_path in args or via --checkpoint argument")
    
    logging.info(f"\nLoading checkpoint from {checkpoint_path}")
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=args["device"][0])
        model._network.load_state_dict(state_dict, strict=True)
        logging.info("Checkpoint loaded successfully with strict=True!")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        logging.error("This usually means the model structure doesn't match the checkpoint.")
        logging.error("Please verify that the config file matches the one used during training.")
        return
    
    # Move model to device and set to eval mode
    model._network.to(model._device)
    model._network.eval()
    
    logging.info("All params: {}".format(count_parameters(model._network)))
    logging.info("Trainable params: {}".format(count_parameters(model._network, True)))
    
    # Perform inference on all classes
    test_dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes), source="test", mode="test"
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args["batch_size"], 
        shuffle=False, 
        num_workers=8
    )
    
    logging.info(f"Performing inference on {model._total_classes} classes...")
    
    # Run evaluation
    model.test_loader = test_loader
    cnn_accy, nme_accy = model.eval_task()
    
    # Print results
    logging.info("\n" + "="*50)
    logging.info("INFERENCE RESULTS")
    logging.info("="*50)
    
    if nme_accy is not None:
        logging.info("CNN Results: {}".format(cnn_accy["grouped"]))
        logging.info("NME Results: {}".format(nme_accy["grouped"]))
        logging.info("\nCNN Top-1 Accuracy: {:.2f}%".format(cnn_accy["top1"]))
        logging.info("NME Top-1 Accuracy: {:.2f}%".format(nme_accy["top1"]))
    else:
        logging.info("CNN Results: {}".format(cnn_accy["grouped"]))
        logging.info("\nCNN Top-1 Accuracy: {:.2f}%".format(cnn_accy["top1"]))
    
    logging.info("="*50)
    
    return cnn_accy, nme_accy


def _set_device(args):
    device_type = args["device"]
    gpus = []
    
    for device in device_type:
        # Handle both string and int device IDs
        if isinstance(device, str):
            device = int(device)
        
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID (can be comma-separated for multiple GPUs)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override config with command line arguments
    config["checkpoint_path"] = args.checkpoint
    
    # Parse device argument
    if args.device:
        device_list = [d.strip() for d in args.device.split(',')]
        # Convert to int, handling -1 for CPU
        config["device"] = device_list
    
    # Run inference
    inference(config)
