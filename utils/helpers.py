def print_run_arguments(args):

    print("========== Current Operating Parameters ==========")
    for key in sorted(vars(args).keys()):
        print(f"{key}: {getattr(args, key)}")
    print("================================")

