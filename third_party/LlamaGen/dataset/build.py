from dataset.imagenet_code import build_imagenet_code


def build_dataset(args, **kwargs):
    if args.dataset == "imagenet_code":
        return build_imagenet_code(args)
    raise ValueError(f"dataset {args.dataset} is not supported by this experiment")
