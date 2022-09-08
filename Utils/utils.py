import tfx
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
import os
from tfx.proto import example_gen_pb2
from tfx.v1 import proto
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
import tensorflow as tf
import yaml


def create_pipline(pipeline_name, pipeline_root, data_root,
                   serving_model_dir, metadata_path,
                   _train_module_file, _tuner_module_file, _transform_module_file, first_time_tuning=True):
    components = []  # Initiating empty list to append the components to it after creating eatch

    output = proto.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            proto.SplitConfig.Split(name="train", hash_buckets=8),
            proto.SplitConfig.Split(name="eval", hash_buckets=1),
            proto.SplitConfig.Split(name="test", hash_buckets=1)
        ])
    )
    example_gen = CsvExampleGen(input_base=data_root, output_config=output)
    components.append(example_gen)

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False)
    components.append(schema_gen)

    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=_transform_module_file)
    components.append(transform)

    if first_time_tuning:
        tuner = tfx.components.Tuner(
            module_file=_tuner_module_file,  # Contains `tuner_fn`.
            examples=transform.outputs['transformed_examples'],
            transform_graph=transform.outputs['transform_graph'],
            train_args=trainer_pb2.TrainArgs(num_steps=20),
            eval_args=trainer_pb2.EvalArgs(num_steps=5))
        components.append(tuner)

        trainer = tfx.components.Trainer(
            module_file=_train_module_file,
            examples=transform.outputs['transformed_examples'],
            transform_graph=transform.outputs['transform_graph'],
            hyperparameters=tuner.outputs['best_hyperparameters'],
            schema=schema_gen.outputs['schema'],
            train_args=tfx.proto.TrainArgs(num_steps=100),
            eval_args=tfx.proto.EvalArgs(num_steps=5))
        components.append(trainer)
    else:
        # TODO adjust it later
        hparams_importer = ImporterNode(
            instance_name='import_hparams',
            # This can be Tuner's output file or manually edited file. The file contains
            # text format of hyperparameters (kerastuner.HyperParameters.get_config())
            source_uri='path/to/best_hyperparameters.txt',
            artifact_type=HyperParameters)

        trainer = tfx.components.Trainer(
            module_file=_train_module_file,
            examples=transform.outputs['transformed_examples'],
            transform_graph=transform.outputs['transform_graph'],
            hyperparameters=hparams_importer.outputs['result'],
            schema=schema_gen.outputs['schema'],
            train_args=tfx.proto.TrainArgs(num_steps=100),
            eval_args=tfx.proto.EvalArgs(num_steps=5))
        components.append(trainer)

    # TODO put model evaluation here

    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))
    components.append(pusher)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata
        .sqlite_metadata_connection_config(metadata_path),
        components=components)
