from setuptools import setup, Extension

ext_modules = [
    Extension(
        "cpp_jsonl_fastscan",
        sources=["cpp_jsonl_fastscan.cpp"],
        language="c++",
        extra_compile_args=["/O2"] if __import__('sys').platform.startswith('win') else ["-O3", "-std=c++17"],
    )
]

setup(
    name="cpp_jsonl_fastscan",
    version="0.1.0",
    description="C++ binding for fast JSONL scanning",
    ext_modules=ext_modules,
)
