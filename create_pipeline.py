






import tensorflow as tf
from tfx import v1 as tfx


import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                      module_file: str, serving_model_dir: str,
                     metadata_path: str) -> tfx.dsl.Pipeline:
  """Implements the penguin pipeline with TFX."""
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=data_root)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(
      examples=example_gen.outputs['examples'])

  schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'])

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # NEW: Transforms input data using preprocessing_fn in the 'module_file'.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      materialize=False,
      module_file=module_file)

  # Uses user-provided Python function that trains a model.
  trainer = Trainer(
      module_file=module_file,
      examples=example_gen.outputs['examples'],

      # NEW: Pass transform_graph to the trainer.
      transform_graph=transform.outputs['transform_graph'],

      train_args=tfx.proto.TrainArgs(num_steps=1000),
      eval_args=tfx.proto.EvalArgs(num_steps=200))

#  # Get the latest blessed model for model validation.
#   model_resolver = tfx.dsl.Resolver(
#       strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
#       model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
#       model_blessing=tfx.dsl.Channel(
#           type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
#               'latest_blessed_model_resolver')

#   # Uses TFMA to compute evaluation statistics over features of a model and
#   # perform quality validation of a candidate model (compared to a baseline).
#   eval_config = tfma.EvalConfig(
#       model_specs=[tfma.ModelSpec(label_key='sentiment')],
#       slicing_specs=[tfma.SlicingSpec()],
#       metrics_specs=[
#           tfma.MetricsSpec(metrics=[
#               tfma.MetricConfig(
#                   class_name='BinaryAccuracy',
#                   threshold=tfma.MetricThreshold(
#                       value_threshold=tfma.GenericValueThreshold(
#                           # Increase this threshold when training on complete
#                           # dataset.
#                           lower_bound={'value': 0.01}),
#                       # Change threshold will be ignored if there is no
#                       # baseline model resolved from MLMD (first run).
#                       change_threshold=tfma.GenericChangeThreshold(
#                           direction=tfma.MetricDirection.HIGHER_IS_BETTER,
#                           absolute={'value': -1e-2})))
#           ])
#       ])

#   evaluator = Evaluator(
#       examples=example_gen.outputs['examples'],
#       model=trainer.outputs['model'],
#       baseline_model=model_resolver.outputs['model'],
#       eval_config=eval_config)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
    #   model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  components = [
      example_gen,
      statistics_gen,
      schema_gen,
      example_validator,
      transform,
      trainer,
    #   model_resolver,
    #   evaluator,
      pusher,
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
    #   enable_cache=True,
      components=components)