def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.dataset:<20}{"Data Path:":<20}{args.data_path:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.device:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Train Epochs:":<20}{args.epochs:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Batch Size:":<20}{args.batch_size:<20}{"Patience:":<20}{args.patience:<20}')
    print(f'  {"Learning Rate:":<20}{args.learning_rate:<20}{"Num Workers:":<20}{args.num_workers:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Seed:":<20}{args.seed:<20}')
    print()

    print("\033[1m" + "STFT&DE" + "\033[0m")
    print(f'  {"FS:":<20}{args.fs:<20}{"N Fft:":<20}{args.n_fft:<20}')
    print(f'  {"Hop Length:":<20}{args.hop_length:<20}{"Win Length:":<20}{args.win_length:<20}')
    print()