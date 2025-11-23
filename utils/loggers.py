from tensorboardX import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if hasattr(v, "item"):
                v = v.item()
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class WandbLogger:
    def __init__(self, args):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError as exc:
            raise ImportError(
                "要使用Weights and Biases记录器，请安装wandb。"
                "运行 `pip install wandb` 来安装它。"
            ) from exc

        if self._wandb.run is None:
            self._wandb.init(project=args.project, config=args)

    def log_epoch_metrics(self, metrics, commit=True):
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'test' in k:
                self._wandb.log({f'Global Test/{k}': v}, commit=False)

        self._wandb.log({}, commit=commit)

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )
        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')


def show_config(**kwargs):
    print("Configurations:")
    print("-" * 70)
    print("|%25s | %40s|" % ("keys", "values"))
    print("-" * 70)
    for key, value in kwargs.items():
        print("|%25s | %40s|" % (str(key), str(value)))
    print("-" * 70)
