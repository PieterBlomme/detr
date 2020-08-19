from dataclasses import dataclass
from typing import Optional, Tuple

from rvai.base.cell import CellMode, TrainableCell, cell
from rvai.base.context import (
    InferenceContext,
    ModelContext,
    ParameterContext,
    TestContext,
    TrainingContext,
)
from rvai.base.data import (
    Annotations,
    Class,
    Dataset,
    DatasetConfig,
    Example,
    Expertise,
    Inputs,
    Measurement,
    Measurements,
    Metrics,
    Outputs,
    Parameters,
    Samples,
    State,
    Tag,
)
from rvai.base.test import TestSession
from rvai.base.training import Model, ModelConfig, ModelPath, TrainingSession


@dataclass
class MyInputs(Inputs):
    # TODO: describe the Cell's input types
    # Example:
    # image: Image = Inputs.field(
    #     name="Image", description="An image."
    # )
    pass


@dataclass
class MyOutputs(Outputs):
    # TODO: describe the Cell's output types
    # Example:
    # detections: List[BoundingBox] = Outputs.field(
    #     name="Detections", description="Detected objects."
    # )
    pass


@dataclass
class MyParameters(Parameters):
    # TODO: describe the Cell's declared parameter types
    # Example:
    # threshold: FloatRange = Parameters.field(
    #     name="Detections", description="Detected objects.",
    #     default=FloatRange(value=0.5, min=0.0, max=1.0)
    # )
    pass


@dataclass
class MyProcessedParameters(Parameters):
    # TODO: describe the Cell's processed parameter types
    pass


@dataclass
class MyState(State):
    # OPTIONAL: declare state by setting this class
    # on the @cell decorator, for example `@cell(state=MyState)`

    # TODO: describe the Cell's state types
    # Example:
    # counter: Integer = State.field(
    #     default=Integer(0),
    #     name="Counter", description="Frame counter."
    # )
    pass


@dataclass
class MyMeasurement(Measurement):
    # TODO: add `Measurement` subclasses for each measurement
    # this cell needs to emit
    # Example:
    # event_type: String = Measurement.field(
    #     name="Event type", description="Type of the event.", index=True
    # )
    # severity: String = Measurement.field(
    #     name="Event severity",
    #     description="Severity of the event.",
    #     options=list(map(String, ["low", "mid", "high"])),
    #     index=True,
    # )
    # info: String = Measurement.field(
    #     name="Event info", description="More info about the event."
    # )
    pass


@dataclass
class MyMeasurements(Measurements):
    # OPTIONAL: declare measurements by setting this class
    # on the @cell decorator, for example `@cell(measurements=MyMeasurements)`

    # TODO: add all `Measurement`s here
    # Example:
    # my_measurement: Type[MyMeasurement]
    pass


@dataclass
class MySamples(Samples):
    # TODO: describe the TrainableCell's sample types
    # Example:
    # image: Image = Samples.field(
    #     name="Image", description="An image."
    # )
    pass


@dataclass
class MyAnnotations(Annotations):
    # TODO: describe the TrainableCell's annotation types
    # Example:
    # detections: List[BoundingBox] = Annotations.field(
    #     name="Detections", description="Detected objects."
    # )
    pass


@dataclass
class MyMetrics(Metrics):
    # TODO: describe the TrainableCell's metric types
    # accuracy: Float = Metrics.field(
    #     name="Accuracy",
    #     short_name="acc",
    #     description="Accuracy",
    #     performance=True # mark as main performance metric
    # )
    pass


@cell
class DetrCell(TrainableCell):

    # =========================================================================
    # Optional: CellBase.process_parameters
    @classmethod
    def process_parameters(
        cls, context: ParameterContext, parameters: MyParameters
    ) -> MyProcessedParameters:
        """Preprocess a Cell's declared parameters."""
        # TODO: implement
        # processed_parameters: MyProcessedParameters = parameters
        # return processed_parameters
        ...

    # =========================================================================

    @classmethod
    def load_model(
        cls,
        context: ModelContext,
        parameters: MyProcessedParameters,
        model_path: Optional[ModelPath],
        dataset_config: Optional[DatasetConfig],
    ) -> Tuple[Model, ModelConfig]:
        """Load a serialized model from disk."""
        # TODO: implement
        ...

    @classmethod
    def train(
        cls,
        context: TrainingContext,
        parameters: MyProcessedParameters,
        model: Model,
        model_config: ModelConfig,
        train_dataset: Dataset[MySamples, MyAnnotations],
        validation_dataset: Dataset[MySamples, MyAnnotations],
        dataset_config: Optional[DatasetConfig],
    ) -> TrainingSession[MyMetrics]:
        """Train a predictive model on annotated data."""
        # TODO: implement
        ...

    # =========================================================================
    # Optional: TrainableCell.test
    @classmethod
    def test(
        cls,
        context: TestContext,
        parameters: MyProcessedParameters,
        model: Model,
        model_config: ModelConfig,
        test_dataset: Dataset[MySamples, MyAnnotations],
        dataset_config: Optional[DatasetConfig],
    ) -> TestSession[MyMetrics]:
        """Test the performance of a predictive model on new, unseen, data.

        This method is optional and can be removed if the algorithm doesn't
        support testing on a test dataset.
        """
        # TODO: implement
        # return MyMetrics(accuracy=Float(0.99))
        ...

    # =========================================================================

    @classmethod
    def predict(
        cls,
        context: InferenceContext,
        parameters: MyProcessedParameters,
        model: Model,
        model_config: ModelConfig,
        inputs: MyInputs,
    ) -> MyOutputs:
        """Make predictions about sampled input data using a predictive model."""
        # TODO: implement
        ...
