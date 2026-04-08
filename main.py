from dataclasses import dataclass
from enum import Enum
import os

from build_dataset import build_dataset
from p_at_100 import benchmark_model
from train_yolo26_seg import train_model

@dataclass
class YearSplit:
    train_years: list[int]
    valid_years: list[int]
    test_years: list[int]

    percent_background_samples_for_train_valid: float
    
    def to_dict(self):
        tr = {year: "train" for year in self.train_years}
        va = {year: "valid" for year in self.valid_years}
        te = {year: "test" for year in self.test_years}
        return {**tr, **va, **te}

@dataclass
class PercentSplit:
    train_percent: float
    valid_percent: float
    test_percent: float

    percent_background_samples_for_train_valid: float

    def to_tuple(self):
        return self.train_percent, self.valid_percent, self.test_percent

@dataclass
class DataParameters:
    split_config: YearSplit | PercentSplit
    augmentation_ratios: dict[int, float]
    max_augments_per_image: int

class ModelSizes(Enum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTRA_LARGE = "extra_large"

@dataclass
class ModelParameters:
    model_size: ModelSizes
    epochs: int
    batch_size: int # Reduce if you hit Out of Memory (OOM) errors on GPU
    patience: int # Early Stopping: If accuracy doesn't improve for x epochs, stop training early to prevent overfitting

def run_full_e2e_process(name: str,
                         model_config: ModelParameters | None = None,
                         input_yml_path: str | None = None,
                         training_config: DataParameters | None = None,
                         trained_model_weights_path: str | None = None):
    assert input_yml_path or training_config, "Must specify either input_yml_path or training_config."
    assert model_config or trained_model_weights_path, "Must specify either model_config or trained_model_weights_path."
    model_input = build_dataset(
        name=name,
        background_samples_percent=training_config.split_config.percent_background_samples_for_train_valid,
        year_train_split=training_config.split_config.to_dict() if isinstance(training_config.split_config, YearSplit) else None,
        train_valid_test_split=training_config.split_config.to_tuple() if isinstance(training_config.split_config, PercentSplit) else None,
        augmentation_ratios=training_config.augmentation_ratios,
        max_augments_per_image=training_config.max_augments_per_image
    ) if input_yml_path is None else input_yml_path

    if not trained_model_weights_path:
        model_directory = train_model(
            name=name,
            model_size=model_config.model_size.value,
            yaml_path=model_input,
            epochs=model_config.epochs,
            patience=model_config.patience,
            batch_size=model_config.batch_size
        ) 
        trained_model_weights_path = os.path.join(model_directory, "weights", "best.pt")
        print(f"Training complete. Best model weights saved at: {trained_model_weights_path}")
    
    benchmark_model(name, trained_model_weights_path)

if __name__ == "__main__":

    # Example usage:
    data_params = DataParameters(
        split_config=YearSplit(
            train_years=[2016,2017,2018,2019],
            valid_years=[2020,2021],
            test_years=[2022,2023],
            percent_background_samples_for_train_valid=0.8
        ),
        augmentation_ratios={
            0: 8,  # bulky waste
            1: 4,  # cardboard
            2: 1.5,  # garbage bag (we want to reduce this class)
            3: 3,  # litter
            4: 12,  # other
        },
        max_augments_per_image=5
    )

    model_params = ModelParameters(
        model_size=ModelSizes.NANO,
        epochs=100,
        batch_size=16,
        patience=20
    )

    run_full_e2e_process(
        name="uw-basic-seg-nano-v1",
        model_config=model_params,
        training_config=data_params
    )