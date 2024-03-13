from super_gradients.import_utils import import_pytorch_quantization_or_install


def main():
    import_pytorch_quantization_or_install()
    import pytorch_quantization

    print(pytorch_quantization.__version__)


if __name__ == "__main__":
    main()
