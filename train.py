import sys
import tempfile
import warnings
from typing import Type

from composer.loggers.logger import LogLevel
from composer.loggers.logger_hparams import WandBLoggerHparams
from composer.trainer import TrainerHparams

from models.deeplabv3_hparams import DeepLabV3Hparams


def warning_on_one_line(message: str,
                        category: Type[Warning],
                        filename: str,
                        lineno: int,
                        file=None,
                        line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    TrainerHparams.register_class(field='model',
                                  register_class=DeepLabV3Hparams,
                                  class_key='my_deeplabv3')
    hparams = TrainerHparams.create(
        cli_args=True)  # reads cli args from sys.argv

    # if using wandb, store the config inside the wandb run
    for logger_hparams in hparams.loggers:
        if isinstance(logger_hparams, WandBLoggerHparams):
            logger_hparams.config = hparams.to_dict()

    trainer = hparams.initialize_object()

    # Log the config to an artifact store
    with tempfile.NamedTemporaryFile(mode="x+") as f:
        f.write(hparams.to_yaml())
        trainer.logger.file_artifact(LogLevel.FIT,
                                     artifact_name="{run_name}/hparams.yaml",
                                     file_path=f.name,
                                     overwrite=True)

    # Print the config to the terminal
    print("*" * 30)
    print("Config:")
    print(hparams.to_yaml())
    print("*" * 30)

    trainer.fit()


if __name__ == "__main__":
    main()