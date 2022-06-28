import os
import pickle
import sys
from enum import Enum

try:
    import tensorrt as trt
except ModuleNotFoundError:
    print('problem importing package: \"tensorrt\" (this will effect execution only if using tensorrt)')


class QuantizationLevel(str, Enum):
    FP32 = 'FP32'
    FP16 = 'FP16'
    INT8 = 'INT8'
    HYBRID = 'Hybrid'

    @staticmethod
    def from_string(quantization_level: str) -> Enum:
        quantization_level = quantization_level.lower()
        if quantization_level == 'fp32':
            return QuantizationLevel.FP32
        elif quantization_level == 'fp16':
            return QuantizationLevel.FP16
        elif quantization_level == 'int8':
            return QuantizationLevel.INT8
        elif quantization_level == 'hybrid':
            return QuantizationLevel.HYBRID
        else:
            raise NotImplementedError(f'Quantization Level: "{quantization_level}" is not supported')


def build_trt_engine_from_onnx_ckpt(onnx_ckpt_path: str,
                                    trt_max_batch_size: int,
                                    quantization_level: QuantizationLevel = QuantizationLevel.FP32) -> trt.ICudaEngine:
    """
    A function for building a trt.ICudaEngine graph from an ONNX model.
    :param onnx_ckpt_path: Path to ONNX model.
    :param quantization_level: The precision to use. Currently supported FP32 and FP16.
    :param trt_max_batch_size: The max batch size allowed for inference.
    :return: An ICudaEngine object (graph of the model).
    """
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)  # VERBOSE for printing
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Fill network attributes with information by parsing model
        with open(onnx_ckpt_path, "rb") as f:
            # Parse model and capture its exit status
            parse_success = parser.parse(f.read())
            # Catch any errors thrown while parsing and exit gracefully on failure
            if not parse_success:
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)

        # Query input names and shapes from parsed TensorRT network
        network_inputs = [network.get_input(i) for i in range(network.num_inputs)]
        input_names = [_input.name for _input in network_inputs]  # ex: ["actual_input1"]
        # Note the original model must have dynamic (-1) dimensions for variable min/opt/max values
        # in the profile dimensions (such as the batch dimension)
        input_shapes = [_input.shape for _input in network_inputs]  # ex: [(-1, 3, 224, 224)]

        # Create optimization profile for dynamic batch dimension
        # Note optimal performance is set for max batch size
        profile0 = builder.create_optimization_profile()
        for name, shape in zip(input_names, input_shapes):
            profile0.set_shape(
                name, min=(1, *shape[1:]), opt=(trt_max_batch_size, *shape[1:]),
                max=(trt_max_batch_size, *shape[1:])
            )
        config.add_optimization_profile(profile0)

        # Additional builder_config flags can be set prior to building the engine
        if quantization_level == QuantizationLevel.FP16:
            config.set_flag(trt.BuilderFlag.FP16)
            try:
                builder.fp16_mode = True
            except:
                # TRT8 breaking API - supporting older versions
                pass
        elif quantization_level == QuantizationLevel.INT8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Compilation parameters
        config.max_workspace_size = 4 << 30
        try:
            builder.max_workspace_size = 4 << 30
        except:
            # TRT8 breaking API - supporting older versions
            pass

        # Set max batch size
        builder.max_batch_size = trt_max_batch_size

        engine = builder.build_engine(network, config)

        return engine


if __name__ == '__main__':
    # from_model = sys.argv[1]
    from_model = 'resnet18_qat.onnx'
    # to_model = sys.argv[2]
    to_model = 'resnet18_qat.fp16.engine'
    # quantization_level = QuantizationLevel.INT8
    quantization_level = QuantizationLevel.FP16
    # quantization_level = QuantizationLevel.FP32

    # quantization_level = QuantizationLevel(sys.argv[3].upper())
    # max_batch_size = int(sys.argv[4])
    max_batch_size = 1
    if not from_model:
        raise ValueError("Please specify an onnx input file as first argument")
    if not to_model:
        raise ValueError("Please specify a trt output file as second argument")
    if os.path.exists(from_model):
        print("Building engine from file {}".format(from_model))
        trt_engine = build_trt_engine_from_onnx_ckpt(onnx_ckpt_path=from_model,
                                                     trt_max_batch_size=max_batch_size,
                                                     quantization_level=quantization_level)
        serialized_engine = bytes(trt_engine.serialize())
        engine_dict = {'engine': serialized_engine,
                       'compiler': 'trt',
                       'precision': quantization_level.value}
        with open(to_model, "wb") as f:
            pickle.dump(engine_dict, f)
            print(
                f'Successfully converted {from_model} to {to_model} with max batch size {max_batch_size} and {quantization_level} quantization.')
    else:
        raise ValueError("The input file does not exist.")