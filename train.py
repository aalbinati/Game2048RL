if __name__ == '__main__':
    import time

    import gym
    import torch
    import torch_ac
    import tensorboardX
    import sys

    import utils
    from model import ACModel
    from env import Game2048

    model_name = "Game2048-v0"
    memory = True
    algorithm = "ppo"
    discount = 0.99
    lr = 0.001
    batch_size = 256
    epochs = 100
    clip_eps = 0.2
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    recurrence = 4
    optim_alpha = 0.99
    optim_eps = 1e-8
    frames = 10000000
    log_interval = 1
    save_interval = 100

    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Set device

    device = torch.device("cpu")
    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(4):
        envs.append(Game2048())  # gym.make(model_name))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(envs[0].observation_space, envs[0].action_space, memory, False)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if algorithm == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, 5, discount, lr, gae_lambda,
                                entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                                optim_alpha, optim_eps, preprocess_obss)
    elif algorithm == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, 128, discount, lr, gae_lambda,
                                entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                                optim_eps, clip_eps, epochs, batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(algorithm))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:mM {:.2f} {:.2f} {:.2f} {:.2f} | F:mM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} |  {:.3f}"
                    .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if save_interval > 0 and update % save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")