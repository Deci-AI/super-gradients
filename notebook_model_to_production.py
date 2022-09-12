from super_gradients.training import models
import torch
import infery

# TODO:
#  There is no "base" notebook so I only wrote the commands and some visualizations that needed to do in Colab.
#  I'll be happy filling it all up when possible deploying (due to input problematic issue)


def main():
    # TODO: State that all SG models can do that same process

    # Load pretrained model
    model = models.get("yolox_s", pretrained_weights="coco")

    # Prepare model for conversion
    model.eval()
    model.prep_model_for_conversion(input_size=[1, 3, 640, 640])

    # Create dummy input for model tracing, with shape [Batch x Channels x Width x Height]
    dummy_input = torch.randn([1, 3, 640, 640])

    # TODO: Visualize input (which won't be dummy input, but an image)

    # Create output path
    onnx_filename = "yolox_s.onnx"

    # Convert model to onnx
    torch.onnx.export(model, dummy_input, onnx_filename)

    # # TODO: Visualize onnx on netron?

    # Load model with infery
    model = infery.load(model_path=onnx_filename, framework_type='onnx', inference_hardware='gpu')

    # Predict
    output = model.predict(dummy_input.numpy())

    # TODO: Visualize output

if __name__ == '__main__':
    main()

