import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="skvmn")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--dim_s", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--size_m", type=int, default=50)
   
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--use_ed", type=int, default=0)
    parser.add_argument("--use_mp", type=int, default=0)
    parser.add_argument("--archi", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--cold_start_len", type=int, default=0, help="If > 0, test only uses first N elements (cold-start evaluation)")

    args = parser.parse_args()

    params = vars(args)
    main(params)
