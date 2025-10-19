#!/usr/bin/env python3

from random import random
from pathlib import Path
import colorsys

import click
import miniacd
import trimesh


def get_formats() -> str:
    formats = trimesh.available_formats()
    return ", ".join(formats)


def random_rgb() -> tuple[int, int, int]:
    return colorsys.hsv_to_rgb(
        random(),
        0.5,
        0.5,
    )


@click.command(
    no_args_is_help=True,
    epilog="""For loading, supported formats are: {}. Output format support
    depends on your trimesh installation.""".format(get_formats()),
)
@click.argument("input-file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-dir",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Directory to save output mesh files",
)
@click.option(
    "--output-format", default="obj", help="File extension for saving part meshes"
)
@click.option(
    "--output-combined",
    default=True,
    help="Output a single mesh file with multiple internal objects",
)
@click.option(
    "--threshold",
    default=0.1,
    type=click.FloatRange(min=0.0),
    help="Concavity threshold at which mesh parts will be accepted",
)
@click.option(
    "--mcts-depth",
    default=3,
    type=click.IntRange(min=0),
    help="Monte Carlo tree search search depth",
)
@click.option(
    "--mcts-iterations",
    default=150,
    type=click.IntRange(min=0),
    help="Monte Carlo tree search search iterations",
)
@click.option(
    "--seed",
    default=42,
    type=click.IntRange(min=0),
    help="Random generator seed for deterministic output",
)
@click.version_option()
def main(
    input_file: str,
    output_format: str,
    output_combined: bool,
    output_dir: str,
    threshold: float,
    mcts_depth: int,
    mcts_iterations: int,
    seed: int,
):
    """
    miniacd decomposes watertight 3D meshes into convex components
    """
    input_mesh = trimesh.load_mesh(input_file)

    # Run the miniacd algorithm.
    mesh = miniacd.Mesh(input_mesh.vertices, input_mesh.faces)
    parts = miniacd.run(
        mesh,
        threshold=threshold,
        max_depth=mcts_depth,
        iterations=mcts_iterations,
        random_seed=seed,
        print=True,
    )

    # Convexify the output slices.
    parts = [part.convex_hull() for part in parts]

    # Build Trimesh objects from the miniacd meshes.
    output_parts = [
        trimesh.Trimesh(
            part.vertices(), part.faces(), vertex_colors=random_rgb(), process=False
        )
        for part in parts
    ]

    # Output a single file of multiple objects or multiple files.
    output_stem = Path(input_file).stem + "_convex"
    if output_combined:
        output_path = Path(output_dir) / f"{output_stem}.{output_format}"
        trimesh.Scene(output_parts).export(output_path)
    else:
        for i, part in enumerate(output_parts):
            output_path = Path(output_dir) / f"{output_stem}_{i}.{output_format}"
            part.export(output_path)


if __name__ == "__main__":
    main(show_default=True, max_content_width=100)
