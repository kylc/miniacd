# miniacd

<p>
    <a href="https://github.com/kylc/miniacd/blob/main/LICENSE-MIT">        <img alt="MIT"    src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
    <a href="https://github.com/kylc/miniacd/blob/main/LICENSE-APACHE">     <img alt="Apache" src="https://img.shields.io/badge/license-Apache-blue.svg"></a>
    <a href="https://github.com/kylc/miniacd/actions/workflows/ci.yaml">    <img alt="ci"     src="https://github.com/kylc/miniacd/actions/workflows/ci.yaml/badge.svg"></a>
    <a href="https://github.com/kylc/miniacd/actions/workflows/python.yaml"><img alt="python" src="https://github.com/kylc/miniacd/actions/workflows/python.yaml/badge.svg"></a>
    <a href="https://pypi.org/project/miniacd/">                            <img alt="PyPI"   src="https://img.shields.io/pypi/v/miniacd.svg"></a>
</p>

**miniacd** decomposes watertight 3D meshes into convex components which aim to be a good approximation of the input shape. It is a compact and high performance implementation of the CoACD algorithm described by Wei et al. and implemented in the [CoACD](https://github.com/SarahWeiii/CoACD) repository.

<img width="2048" height="516" alt="image" src="https://github.com/user-attachments/assets/352be41d-1a93-4dc0-ace8-78a938e5f585" />

## Setup

``` sh
pip install miniacd
```

Or, to install a prerelease version:

1. Download a recent `.whl` from [GitHub Releases](https://github.com/kylc/miniacd/releases)
2. Run `pip install miniacd<...>.whl` (replace `<...>` with the actual filename)
3. Test it: `miniacd --help`

### Building Locally

``` sh
git clone git@github.com:kylc/miniacd.git
cd miniacd

# Build the Rust library
cargo build --release

# OR build a Python wheel
pip wheel .
```

## Usage

You can use the `miniacd` command to process your mesh files. It has wide support for input and output formats, provided by [trimesh](https://trimesh.org/). A typical invocation looks like this:

``` sh
miniacd input_mesh.obj --output-dir output/ --threshold 0.1
```

If you have more specific needs, you can use miniacd as a Python library. See [cli.py](python/miniacd/cli.py) for an example. You can also access the internals by using miniacd as a Rust library.

## References

Xinyue Wei, Minghua Liu, Zhan Ling, and Hao Su. 2022. Approximate convex decomposition for 3D meshes with collision-aware concavity and tree search. ACM Trans. Graph. 41, 4, Article 42 (July 2022), 18 pages. https://doi.org/10.1145/3528223.3530103
