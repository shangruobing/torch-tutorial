import argparse
from dataclasses import dataclass


def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.001)


@dataclass
class Arguments:
    string: str
    device: int
    cpu: bool
    seed: int
    epochs: int
    lr: float
    weight_decay: float


def parser_parse_args(parser) -> Arguments:
    args = parser.parse_args()
    return Arguments(
        string=vars(args),
        device=args.device,
        cpu=args.cpu,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def init_parser_args() -> Arguments:
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    return parser_parse_args(parser)
