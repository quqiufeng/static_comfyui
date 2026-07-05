# Model sampling/prediction types
# Sigma schedules handled by C++ backend (libtorch_std_helper)

class ModelSamplingType:
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    EDM = 5
    CONST = 6
    V_PREDICTION_CONTINUOUS = 7
    FLOW = 8
    X0 = 9


def sampling_type_from_model_type(model_type: int) -> int:
    if model_type == 1:
        return ModelSamplingType.EPS
    elif model_type == 2 or model_type == 3:
        return ModelSamplingType.V_PREDICTION
    elif model_type == 5:
        return ModelSamplingType.EDM
    elif model_type == 6 or model_type == 8:
        return ModelSamplingType.CONST
    elif model_type == 7:
        return ModelSamplingType.V_PREDICTION_CONTINUOUS
    elif model_type == 9 or model_type == 11:
        return ModelSamplingType.X0
    return ModelSamplingType.EPS


def main():
    pass
