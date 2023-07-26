import os
from auxiliary.func_utils \
    import get_args, parse_config, \
    main_cifar10, main_skimage, \
    load_trained_feature_extractor_main

"""
In this script I implement the interaction with CLIP functions using CIFAR10 as a first stage before 
implementations using a different feature extractor.

This is the most recent implementation in this folder.
"""

if __name__ == "__main__":

    args = get_args()
    config = parse_config(args.config)
    weights_dir = os.path.join(config["results_dir"], "training_results")
    num_features = config["num_features"]
    model_name = config["model_name_str"]
    read_epoch_num = config["read_epoch_num"]

    choose_main = config["choose_main"]

    if choose_main == "cifar10_clip":
        main_cifar10()
        print("Done comparing CIFAR10")
    elif choose_main == "skimage":
        main_skimage()
        print("Done comparing skimage")
    elif choose_main == "load_trained_demo":
        # loading trained model:
        trained_feature_extractor = load_trained_feature_extractor_main(model_name=model_name, num_features=num_features,
                                                                        epoch_num=read_epoch_num, weights_dir=weights_dir)
        main_cifar10(feature_extractor=trained_feature_extractor)
    else:
        raise ValueError("No action chosen in the configuration file.")

    print("Done :)")

