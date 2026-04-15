import os
import argparse
import json
import copy
from pykt.config import que_type_models
import torch
torch.set_num_threads(2)

from pykt.models import evaluate_splitpred_question, load_model, lpkt_evaluate_multi_ahead, init_model, evaluate
from pykt.datasets import init_test_datasets
from pykt.utils import set_seed

def main(params):
    if params['use_wandb'] ==1:
        import wandb
        wandb.init()
    save_dir, use_pred, ratio = params["save_dir"], params["use_pred"], params["train_ratio"]

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])

        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]

        trained_params = config["params"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
        seq_len = config["train_config"]["seq_len"]
        if model_name in ["saint", "sakt", "atdkt"]:
            model_config["seq_len"] = seq_len
        data_config = config["data_config"]

    print(f"Start evaluating model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")
    
    # Simple test evaluation mode (when use_pred=0 and ratio=0.9)
    if use_pred == 0 and ratio == 0.9:
        print("Running simple test evaluation...")
        
        # Set seed for reproducibility
        set_seed(trained_params["seed"])
        
        # Load data config 
        with open("../configs/data_config.json") as f:
            all_data_config = json.load(f)
        
        # Update the custom dataset path to absolute path
        if dataset_name == "custom":
            abs_data_path = os.path.abspath("../data/custom")
            all_data_config["custom"]["dpath"] = abs_data_path
            data_config["dpath"] = abs_data_path
        
        # Prepare data config for init_test_datasets  
        test_data_config = all_data_config[dataset_name].copy()
        test_data_config["dataset_name"] = dataset_name
        
        # Use the test loader from init_test_datasets
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(test_data_config, model_name, batch_size=256)
        
        print(f"Test dataset loaded with {len(test_loader)} batches")
        
        # Initialize model
        print("Initializing model...")
        model = init_model(model_name, model_config=model_config, data_config=all_data_config[dataset_name], emb_type=emb_type)
        
        # Load trained weights
        print("Loading trained model weights...")
        model_path = os.path.join(save_dir, f"{emb_type}_model.ckpt")
        net = torch.load(model_path, map_location="cpu")
        model.load_state_dict(net)
        
        # Set to evaluation mode
        model.eval()
        print("Model loaded successfully!")
        
        # Evaluate model
        print("Evaluating model performance...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Run evaluation
        testauc, testacc = evaluate(model, test_loader, model_name, device)
        
        print("\\n" + "=" * 40)
        print("EVALUATION RESULTS")
        print("=" * 40)
        print(f"Model: {model_name.upper()}")
        print(f"Dataset: {dataset_name}")
        print(f"Test AUC: {testauc:.4f}")
        print(f"Test Accuracy: {testacc:.4f}")
        print("=" * 40)
        
        dfinal = {
            "testauc": testauc,
            "testacc": testacc,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "emb_type": emb_type
        }
        
    else:
        # Original complex evaluation logic
        use_pred = True if use_pred == 1 else False
        model = load_model(model_name, model_config, data_config, emb_type, save_dir)

        print(f"Start predict use_pred: {use_pred}, ratio: {ratio}...")
        atkt_pad = True if params["atkt_pad"] == 1 else False
        if model_name == "atkt":
            save_test_path = os.path.join(save_dir, model.emb_type+"_test_ratio"+str(ratio)+"_"+str(use_pred)+"_"+str(atkt_pad)+"_predictions.txt")
        else:
            save_test_path = os.path.join(save_dir, model.emb_type+"_test_ratio"+str(ratio)+"_"+str(use_pred)+"_predictions.txt")
        
        testf = os.path.join(data_config["dpath"], params["test_filename"])
        if model_name in que_type_models and model_name != "lpkt":
            dfinal = model.evaluate_multi_ahead(data_config,batch_size=16,ob_portions=ratio,accumulative=use_pred)
        elif model_name in ["lpkt"]:
            dfinal = lpkt_evaluate_multi_ahead(model, data_config,batch_size=64,ob_portions=ratio,accumulative=use_pred)
        else:
            dfinal = evaluate_splitpred_question(model, data_config, testf, model_name, save_test_path, use_pred, ratio, atkt_pad)
    
    for key in dfinal:
        print(key, dfinal[key])
    dfinal.update(config["params"])
    if params['use_wandb'] ==1:
        wandb.log(dfinal)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--test_filename", type=str, default="test.csv")
    parser.add_argument("--use_pred", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--atkt_pad", type=int, default=0)
    parser.add_argument("--use_wandb", type=int, default=1)

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
