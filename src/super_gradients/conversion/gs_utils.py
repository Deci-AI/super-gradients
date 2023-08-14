def import_onnx_graphsurgeon_or_fail_with_instructions():
    try:
        import onnx_graphsurgeon as gs
    except ImportError:
        raise ImportError(
            "onnx-graphsurgeon is required to use export API. "
            "Please install it with pip install onnx_graphsurgeon==0.3.27 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    return gs


def import_onnx_graphsurgeon_or_install():
    try:
        import onnx_graphsurgeon as gs

        return gs
    except ImportError:
        import pip

        pip.main(["install", "onnx_graphsurgeon==0.3.27", "--extra-index-url", "https://pypi.ngc.nvidia.com"])

        return import_onnx_graphsurgeon_or_fail_with_instructions()
