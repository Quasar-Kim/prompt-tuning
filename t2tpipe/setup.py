import dataclasses
from copy import copy
from typing import Any, Dict, List, Union, overload

from t2tpipe.dataclass import Env, Model, Slot, Task
from t2tpipe.datamodule import T2tPipeDataModule
from t2tpipe.datapipe import TransformDataPipe
from t2tpipe.postprocessor import PostProcessor


def setup(
    *,
    model: Model,
    task: Task,
    runtime_config: Dict[str, Any],
    prediction: bool = False,
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
    env = _setup_slots(env)
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
    if env.task.postprocessors is None:
        env = dataclasses.replace(
            env, task=dataclasses.replace(env.task, postprocessors=[])
        )
    if env.task.metrics is None:
        env = dataclasses.replace(env, task=dataclasses.replace(env.task, metrics=[]))
    if env.task.metric_preprocessors is None:
        env = dataclasses.replace(
            env, task=dataclasses.replace(env.task, metric_preprocessors=[])
        )
    if env.model.pipes is None:
        env = dataclasses.replace(env, model=dataclasses.replace(env.model, pipes={}))
    if env.model.postprocessors is None:
        env = dataclasses.replace(
            env, model=dataclasses.replace(env.model, postprocessors={})
        )
    return env


def _setup_slots(env: Env) -> Env:
    assert env.model.pipes is not None
    pipes = _insert_model_pipes_to_slots(env.task.pipes, env.model.pipes)
    assert env.task.postprocessors is not None
    assert env.model.postprocessors is not None
    postprocessors = _insert_model_pipes_to_slots(
        env.task.postprocessors, env.model.postprocessors
    )

    env = dataclasses.replace(
        env,
        task=dataclasses.replace(env.task, pipes=pipes, postprocessors=postprocessors),
    )
    return env


def _insert_model_pipes_to_slots(pipes: List, inserted_pipes: Dict[str, Any]) -> List:
    pipe_without_slots = []
    for pipe in pipes:
        if isinstance(pipe, Slot):
            try:
                inserted_pipe = inserted_pipes[pipe.name]
                pipe_without_slots.append(inserted_pipe)
            except KeyError:
                if pipe.required:
                    raise ValueError(f"Required slot {pipe.name} is not provided.")
                # omit optional slot
        else:
            pipe_without_slots.append(pipe)
    return pipe_without_slots


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
        assert isinstance(pipe, TransformDataPipe)
        pipe.setup(env)

    postprocessors = env.task.postprocessors
    if postprocessors is not None:
        for postprocessor in postprocessors:
            assert isinstance(postprocessor, PostProcessor)
            postprocessor.setup(env)

    metric_preprocessors = env.task.metric_preprocessors
    if metric_preprocessors is not None:
        for processor in metric_preprocessors:
            assert isinstance(processor, PostProcessor)
            processor.setup(env)

    metrics = env.task.metrics
    if metrics is not None:
        for metric in metrics:
            metric.setup(env)


def _setup_datamodule(env: Env):
    env.datamodule.configure(env)
