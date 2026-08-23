import io

from typer.testing import CliRunner

from pragent import __version__
from pragent.cli import _configure_stream_utf8, app


def test_cli_version_uses_package_version():
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"pra {__version__}"


def test_cli_configures_output_stream_as_utf8():
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="gbk")

    _configure_stream_utf8(stream)
    stream.write("数学符号：∈ 𝛼")
    stream.flush()

    assert stream.encoding.lower() == "utf-8"
    assert buffer.getvalue().decode("utf-8") == "数学符号：∈ 𝛼"
