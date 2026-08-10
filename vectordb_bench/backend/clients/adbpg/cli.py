import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class AdbpgTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="Db username", required=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Postgres database password",
            default=lambda: os.environ.get("POSTGRES_PASSWORD", ""),
            show_default="$POSTGRES_PASSWORD",
        ),
    ]
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="Postgres database port",
            default=5432,
            show_default=True,
            required=False,
        ),
    ]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Db name", required=True)]
    hnsw_m: Annotated[
        int,
        click.option("--hnsw-m", type=int, help="hnsw_m", default=48, show_default=True, required=False),
    ]
    ef_search: Annotated[
        int,
        click.option("--ef-search", type=int, help="ef_search", default=150, show_default=True, required=False),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            help="ef_construction",
            default=600,
            show_default=True,
            required=False,
        ),
    ]
    nlist: Annotated[
        int,
        click.option("--nlist", type=int, help="nlist", default=1024, show_default=True, required=False),
    ]
    rabitq_bits: Annotated[
        int,
        click.option("--rabitq-bits", type=int, help="rabitq_bits", default=7, show_default=True, required=False),
    ]
    quantize_rescore_amp: Annotated[
        float,
        click.option(
            "--quantize-rescore-amp",
            type=float,
            help="fastann.quantize_rescore_amp",
            default=0.0,
            show_default=True,
            required=False,
        ),
    ]
    nova_adaptive_gamma: Annotated[
        float,
        click.option(
            "--nova-adaptive-gamma",
            type=float,
            help="fastann.nova_adaptive_gamma",
            default=0.0,
            show_default=True,
            required=False,
        ),
    ]
    auto_reduction: Annotated[
        bool,
        click.option(
            "--auto-reduction/--no-auto-reduction",
            "auto_reduction",
            type=bool,
            help="Index WITH auto_reduction=on when enabled",
            default=False,
            show_default=True,
            required=False,
        ),
    ]
    max_scan_points: Annotated[
        int,
        click.option(
            "--max-scan-points",
            type=int,
            help="max_scan_points",
            default=20000,
            show_default=True,
            required=False,
        ),
    ]
    index_scan_mode: Annotated[
        str,
        click.option(
            "--index-scan-mode",
            type=str,
            help="fastann.index_scan_mode",
            default="snapshot",
            show_default=True,
            required=False,
        ),
    ]
    algorithm: Annotated[
        str,
        click.option(
            "--algorithm",
            type=str,
            help="algorithm",
            default="novamr",
            show_default=True,
            required=False,
        ),
    ]
    build_parallel_processes: Annotated[
        int,
        click.option(
            "--build-parallel-processes",
            type=int,
            help="Sets the maximum process to build index",
            required=False,
        ),
    ]
    pca_dim: Annotated[
        int | None,
        click.option(
            "--pca-dim",
            type=int,
            help="PCA dimension for index dimensionality reduction",
            default=None,
            show_default=True,
            required=False,
        ),
    ]
    nprobe: Annotated[
        int,
        click.option(
            "--nprobe",
            type=int,
            help="fastann.nova_nprobe (novad search)",
            default=5,
            show_default=True,
            required=False,
        ),
    ]
    topk_amp: Annotated[
        int,
        click.option(
            "--topk-amp",
            type=int,
            help="fastann.topk_amp: novad IVF candidate amplification factor",
            default=10,
            show_default=True,
            required=False,
        ),
    ]
    nova_build_optimize_level: Annotated[
        int,
        click.option(
            "--nova-build-optimize-level",
            type=int,
            help=(
                "fastann.nova_build_optimize_level: 3=THP+graph opt "
                "(needs /mnt/nova_thp), 2=graph opt only (local dev)"
            ),
            default=3,
            show_default=True,
            required=False,
        ),
    ]
    hybrid_mode: Annotated[
        str,
        click.option(
            "--hybrid-mode",
            type=click.Choice(["none", "join", "unified"]),
            help="Hybrid-search schema flavor",
            default="none",
            show_default=True,
            required=False,
        ),
    ]
    hybrid_storage: Annotated[
        str,
        click.option(
            "--hybrid-storage",
            type=click.Choice(["distributed", "master_only"]),
            help="Physical placement for hybrid tables",
            default="distributed",
            show_default=True,
            required=False,
        ),
    ]
    nova_topk_amp_mul: Annotated[
        float,
        click.option(
            "--nova-topk-amp-mul",
            type=float,
            help="fastann.nova_topk_amp_mul (bitmap push-down path scan amplification multiplier)",
            default=1.0,
            show_default=True,
            required=False,
        ),
    ]
    nova_topk_amp_add: Annotated[
        float,
        click.option(
            "--nova-topk-amp-add",
            type=float,
            help="fastann.nova_topk_amp_add (bitmap push-down path scan amplification addend)",
            default=0.0,
            show_default=True,
            required=False,
        ),
    ]
    force_plan: Annotated[
        str,
        click.option(
            "--force-plan",
            type=click.Choice(["auto", "brute_force", "bitmap_pushdown", "expression", "post_filter"]),
            help=(
                "Force one of the 4 verified hybrid plan recipes (validated across 5 "
                "selectivity rates on 1M and 10M). auto = let planner decide."
            ),
            default="auto",
            show_default=True,
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(AdbpgTypedDict)
def AdbpgNova(**parameters: Unpack[AdbpgTypedDict]):
    from .config import AdbpgConfig, AdbpgIndexConfig

    if parameters["hybrid_storage"] == "master_only" and parameters["load"]:
        raise click.UsageError(
            "master-only data preparation must use "
            "scripts/prepare_hybrid_master_only.py; run the benchmark with "
            "--skip-load --skip-drop-old",
        )

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.Adbpg,
        db_config=AdbpgConfig(
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=AdbpgIndexConfig(
            hnsw_m=parameters["hnsw_m"],
            ef_search=parameters["ef_search"],
            ef_construction=parameters["ef_construction"],
            nlist=parameters["nlist"],
            algorithm=parameters["algorithm"],
            build_parallel_processes=parameters["build_parallel_processes"],
            rabitq_bits=parameters["rabitq_bits"],
            quantize_rescore_amp=parameters["quantize_rescore_amp"],
            nova_adaptive_gamma=parameters["nova_adaptive_gamma"],
            auto_reduction=parameters["auto_reduction"],
            pca_dim=parameters["pca_dim"],
            max_scan_points=parameters["max_scan_points"],
            index_scan_mode=parameters["index_scan_mode"],
            nprobe=parameters["nprobe"],
            topk_amp=parameters["topk_amp"],
            nova_build_optimize_level=parameters["nova_build_optimize_level"],
            hybrid_mode=parameters["hybrid_mode"],
            hybrid_storage=parameters["hybrid_storage"],
            nova_topk_amp_mul=parameters["nova_topk_amp_mul"],
            nova_topk_amp_add=parameters["nova_topk_amp_add"],
            force_plan=parameters["force_plan"],
        ),
        **parameters,
    )
