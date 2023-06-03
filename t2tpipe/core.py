from typing import Dict, Any
import dataclasses

from t2tpipe.dataclass import Task, Model, Env
from t2tpipe.datamodule import T2tPipeDataModule
from t2tpipe.postprocessor import NoopPostProcessor


def setup(
    *,
    model: Model,
    task: Task,
    runtime_config: Dict[str, Any],
    prediction: bool = False
):
    env = Env(
        model=model,
        task=task,
        datamodule=T2tPipeDataModule(),
        runtime_config=runtime_config,
        prediction=prediction,
        pad_to=task.pad_to,
    )
    env = _set_defaults(env)
    _setup_model(env)
    _setup_task(env)
    _setup_datamodule(env)
    return env.model.module, env.datamodule


def _set_defaults(env: Env) -> Env:
    if env.model.padder is None:
        default_padder = env.model.feature_converter.get_default_padder()
        if default_padder is not None:
            env = dataclasses.replace(
                env, model=dataclasses.replace(env.model, padder=default_padder)
            )
    if env.task.postprocessor is None:
        noop_postprocessor = NoopPostProcessor()
        env = dataclasses.replace(
            env, task=dataclasses.replace(env.task, postprocessor=noop_postprocessor)
        )
    if env.task.metrics is None:
        env = dataclasses.replace(env, task=dataclasses.replace(env.task, metrics=[]))
    return env


def _setup_model(env: Env):
    env.model.feature_converter.setup(env)
    env.model.module.configure(env)

    padder = env.model.padder
    if padder is not None:
        padder.setup(env)


def _setup_task(env: Env):
    for src in env.task.source.values():
        src.setup(env)

    for pipe in env.task.pipes:
        pipe.setup(env)

    postprocessor = env.task.postprocessor
    if postprocessor is not None:
        postprocessor.setup(env)

    metrics = env.task.metrics
    if metrics is not None:
        for metric in metrics:
            metric.setup(env)


def _setup_datamodule(env: Env):
    env.datamodule.configure(env)
