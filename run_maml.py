import argparse
from maml import MAML
from utils import dotdict
from models.cnn_model import CNNModel


def build_network(args):
    hparams = dotdict(
        {
            "dim_output": args.num_classes,
            "inner_update_lr": args.inner_update_lr,
            "meta_lr": args.meta_lr,
            "meta_test_num_inner_updates": args.meta_test_num_inner_updates,
            "dim_hidden": args.dim_hidden,
            "img_size": 28,
            "channels": 1,
        }
    )
    model = CNNModel(hparams)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, help="Number of samples per class")
    parser.add_argument("--num-samples-per-class", type=int, help="Number of samples per class")
    parser.add_argument("--data-folder", help="path ot omniglot folder")
    parser.add_argument("--batch-size", type=int, help="Batch size: This is equal to the\
                                        number of tasks per episode")
    parser.add_argument("--inner-update-lr", default=0.4, type=float, help="Learning rate for the inner update")
    parser.add_argument("--meta-lr", type=float, default=0.001, help="Learning rate for the meta learner")
    parser.add_argument("--num-meta-train-iterations", type=int, default=1000, help="Number pf meta training iterations")
    parser.add_argument("--num-inner-updates", type=int, default=1, help="Number of inner gradient steps, during train time")
    parser.add_argument("--meta-test-num-inner-updates", type=int, default=1, help="Number of inner gradient steps during meta test time")
    parser.add_argument("--dim-hidden", type=int, default=16, help="Number of convlution filters")
    parser.add_argument("--num-meta-test-classes", type=int, help="Number of classes in meta test time")
    parser.add_argument("--num-meta-test-samples-per-class", type=int, help="Number of samples per class, during test time")
    parser.add_argument("--num-meta-validation-iterations", type=int, help="Number of epsiodes for validation.")
    parser.add_argument("--num-meta-test-iterations", type=int, help="Number of iterations during meta test time")
    parser.add_argument("--validation-frequency", type=int, dest="Validation Frequency")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = build_network(args)
    model.to(args.device)
    maml = MAML(args, model)
    maml.train()
    maml.test()


if __name__ == "__main__":
    main()
