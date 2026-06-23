"""Wrapper around the Aliyun ADBPG (AnalyticDB for PostgreSQL) vector database."""

import json
import logging
import os
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from . import hybrid_synth
from .config import AdbpgConfigDict, AdbpgIndexConfig

log = logging.getLogger(__name__)


class Adbpg(VectorDB):
    """ADBPG vector database client, using psycopg."""

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
        FilterOp.ArrayContains,
        FilterOp.JoinArrayOverlap,
    ]

    conn: psycopg.Connection[Any] | None = None
    cursor: psycopg.Cursor[Any] | None = None

    _search: sql.Composed

    def __init__(
        self,
        dim: int,
        db_config: AdbpgConfigDict,
        db_case_config: AdbpgIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "Adbpg"
        self.case_config = db_case_config
        # Allow the framework layer (task_runner) to inject a case-specific table
        # name via the `collection_name` kwarg (see Doris for the same pattern).
        override_name = kwargs.get("collection_name")
        self.table_name = override_name if override_name else db_config["table_name"]
        self.connect_config = db_config["connect_config"]
        self.dim = dim
        self.with_scalar_labels = with_scalar_labels

        self._primary_field = "id"
        self._vector_field = "embedding"
        self._scalar_label_field = "label"
        # Index name derives from the table name + algorithm, e.g. vector_1024d_10m_novam_index.
        self._index_name = f"{self.table_name}_{self.case_config.algorithm}_index"

        self.where_clause = ""

        # Hybrid-search state: schema flavor and EXPLAIN ANALYZE diagnostic buffer.
        self.hybrid_mode = getattr(self.case_config, "hybrid_mode", "none")
        self.explain_plans: list[dict] = []
        self._explain_count = 0

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        log.info(f"{self.name} config values: {self.connect_config}\n{self.case_config}")
        if not any(
            (
                self.case_config.create_index_before_load,
                self.case_config.create_index_after_load,
            ),
        ):
            msg = (
                f"{self.name} config must create an index using create_index_before_load or create_index_after_load"
                f"{self.name} config values: {self.connect_config}\n{self.case_config}"
            )
            log.error(msg)
            raise RuntimeError(msg)

        if drop_old:
            self._drop_index()
            self._drop_table()
            if self.hybrid_mode == "join":
                self._drop_doc_table()
            self._create_table(dim)
            self._create_hybrid_aux_indexes()
            if self.case_config.create_index_before_load:
                self._create_index()

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)
        register_vector(conn)
        conn.autocommit = False
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    def _generate_search_query(self) -> sql.Composed:
        search_param = self.case_config.search_param()
        distance_operator = {
            "l2": "<->",
            "ip": "<#>",
            "cosine": "<=>",
        }.get(search_param["metric"], "<->")

        where_clause = sql.SQL(self.where_clause) if self.where_clause else sql.SQL("")

        return sql.Composed(
            [
                sql.SQL(
                    "SELECT {primary_field} FROM public.{table_name} {where_clause} ORDER BY {vector_field} ",
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    primary_field=sql.Identifier(self._primary_field),
                    where_clause=where_clause,
                    vector_field=sql.Identifier(self._vector_field),
                ),
                sql.SQL(distance_operator),
                sql.SQL(" {search_vector}::vector({dim}) LIMIT %s::int").format(
                    search_vector=sql.Placeholder(),
                    dim=self.dim,
                ),
            ],
        )

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """Open a session, apply GUCs, yield, then close."""
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        session_options: Sequence[dict[str, Any]] = self.case_config.session_param()["session_options"]

        if len(session_options) > 0:
            for setting in session_options:
                command = sql.SQL("SET {setting_name} = {val};").format(
                    setting_name=sql.Identifier(setting["parameter"]["setting_name"]),
                    val=sql.Identifier(str(setting["parameter"]["val"])),
                )
                log.debug(command.as_string(self.cursor))
                self.cursor.execute(command)
            self.conn.commit()

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table : {self.table_name}")

        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{table_name}").format(
                table_name=sql.Identifier(self.table_name),
            ),
        )
        self.conn.commit()

    def optimize(self, data_size: int | None = None):
        if self.hybrid_mode == "join":
            self._populate_doc_table()
        self._post_insert()

    def _populate_doc_table(self):
        assert self.conn is not None
        assert self.cursor is not None
        doc = self.case_config.doc_table_name
        join_field = self.case_config.doc_join_field
        tags_field = self.case_config.doc_tags_field
        log.info(f"{self.name} populate doc table {doc}")
        # Clear and refill so reruns are idempotent.
        self.cursor.execute(sql.SQL("TRUNCATE TABLE public.{doc}").format(doc=sql.Identifier(doc)))
        rows = self.cursor.execute(
            sql.SQL("SELECT DISTINCT {col} FROM public.{tbl}").format(
                col=sql.Identifier(join_field),
                tbl=sql.Identifier(self.table_name),
            ),
        ).fetchall()
        with self.cursor.copy(
            sql.SQL("COPY public.{doc} ({jf}, {tf}) FROM STDIN (FORMAT BINARY)").format(
                doc=sql.Identifier(doc),
                jf=sql.Identifier(join_field),
                tf=sql.Identifier(tags_field),
            ),
        ) as copy:
            copy.set_types(["bigint", 1009])
            for r in rows:
                doc_id = int(r[0])
                copy.write_row((doc_id, hybrid_synth.doc_tags_for(doc_id)))
        self.conn.commit()

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()
        self._analyze_tables()

    def _analyze_tables(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        tables = [self.table_name]
        if self.hybrid_mode == "join":
            tables.append(self.case_config.doc_table_name)
        for tbl in tables:
            log.info(f"{self.name} analyze table : {tbl}")
            self.cursor.execute(sql.SQL("ANALYZE {tbl}").format(tbl=sql.Identifier(tbl)))
            self.conn.commit()

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop index : {self._index_name}")

        drop_index_sql = sql.SQL("DROP INDEX IF EXISTS {index_name}").format(
            index_name=sql.Identifier(self._index_name),
        )
        log.debug(drop_index_sql.as_string(self.cursor))
        self.cursor.execute(drop_index_sql)
        self.conn.commit()

    def _set_parallel_index_build_param(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        if index_param["build_parallel_processes"] is not None:
            self.cursor.execute(
                sql.SQL("SET fastann.build_parallel_processes TO {};").format(
                    index_param["build_parallel_processes"],
                ),
            )
            self.conn.commit()

        results = self.cursor.execute(sql.SQL("SHOW fastann.build_parallel_processes;")).fetchall()
        log.info(f"{self.name} parallel index creation parameters: {results}")

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client create index : {self._index_name}")

        index_param = self.case_config.index_param()
        self._set_parallel_index_build_param()

        # Pre-build GUC: raise optimizer level before creating the ANN index.
        # Bit 0 (THP loading) is intentionally cleared because the local
        # adbpg7 instance does not have the sudoers rule needed for the kernel
        # tmpfs mount; bit 1 (graph optimization) is preserved.
        self.cursor.execute(sql.SQL("SET fastann.nova_build_optimize_level = 2;"))
        self.conn.commit()

        options = []
        options.append(sql.SQL("dim = {dim}").format(dim=sql.Literal(self.dim)))
        options.append(
            sql.SQL("distancemeasure = {measure}").format(
                measure=sql.Identifier(index_param["metric"]),
            ),
        )

        for option in index_param["index_creation_with_options"]:
            if option["val"] is not None:
                # When `raw` is set, emit the value as a bare SQL token
                # (e.g. auto_reduction=on) instead of a quoted literal.
                rendered_val = sql.SQL(str(option["val"])) if option.get("raw") else sql.Literal(option["val"])
                options.append(
                    sql.SQL("{option_name} = {val}").format(
                        option_name=sql.Identifier(option["option_name"]),
                        val=rendered_val,
                    ),
                )

        with_clause = sql.SQL("WITH ({});").format(sql.SQL(", ").join(options)) if options else sql.Composed(())

        # Covering index: always INCLUDE the primary field (e.g. id).
        index_create_sql = sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
            USING ann ({vector_field}) INCLUDE ({primary_field})
            """,
        ).format(
            index_name=sql.Identifier(self._index_name),
            table_name=sql.Identifier(self.table_name),
            vector_field=sql.Identifier(self._vector_field),
            primary_field=sql.Identifier(self._primary_field),
        )

        full_sql = (index_create_sql + with_clause).join(" ")
        log.debug(full_sql.as_string(self.cursor))
        self.cursor.execute(full_sql)
        self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            log.info(f"{self.name} client create table : {self.table_name} (hybrid_mode={self.hybrid_mode})")

            extra_columns: list[str] = []
            if self.hybrid_mode == "array":
                # Single-table model: GIN array column + scalar pipeline_id (btree).
                extra_columns.append(f", {self.case_config.array_field} TEXT[]")
                extra_columns.append(", pipeline_id TEXT")
            elif self.hybrid_mode == "join":
                # Two-table model: chunk table carries the pipeline_doc_id FK.
                extra_columns.append(f", {self.case_config.doc_join_field} BIGINT")

            label_column = ""
            if self.with_scalar_labels:
                label_column = f", {self._scalar_label_field} VARCHAR(64)"

            create_sql = sql.SQL(
                f"""
                CREATE TABLE IF NOT EXISTS public.{{table_name}}
                ({{primary_field}} BIGINT PRIMARY KEY, embedding vector({{dim}}){label_column}{''.join(extra_columns)});
                """,
            ).format(
                table_name=sql.Identifier(self.table_name),
                primary_field=sql.Identifier(self._primary_field),
                dim=dim,
            )
            self.cursor.execute(create_sql)

            self.cursor.execute(
                sql.SQL(
                    "ALTER TABLE public.{table_name} ALTER COLUMN embedding SET STORAGE PLAIN;",
                ).format(table_name=sql.Identifier(self.table_name)),
            )

            if self.hybrid_mode == "join":
                self._create_doc_table()

            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create adbpg table: {self.table_name} error: {e}")
            raise e from None

    def _create_doc_table(self):
        """Create the doc-side table for the 2-table JOIN scenario."""
        assert self.cursor is not None
        doc = self.case_config.doc_table_name
        join_field = self.case_config.doc_join_field
        tags_field = self.case_config.doc_tags_field
        log.info(f"{self.name} create doc table {doc}")
        self.cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS public.{doc} ({join_field} BIGINT PRIMARY KEY, {tags_field} TEXT[])",
            ).format(
                doc=sql.Identifier(doc),
                join_field=sql.Identifier(join_field),
                tags_field=sql.Identifier(tags_field),
            ),
        )

    def _drop_doc_table(self):
        assert self.cursor is not None
        doc = self.case_config.doc_table_name
        log.info(f"{self.name} drop doc table {doc}")
        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{doc}").format(doc=sql.Identifier(doc)),
        )
        self.conn.commit()

    def _create_hybrid_aux_indexes(self):
        """Create GIN indexes on hybrid columns (one-time, after table create, before load)."""
        if self.hybrid_mode == "none":
            return
        assert self.conn is not None
        assert self.cursor is not None
        if self.hybrid_mode == "array":
            self.cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON public.{tbl} USING gin ({col})",
                ).format(
                    idx=sql.Identifier(f"{self.table_name}_{self.case_config.array_field}_gin"),
                    tbl=sql.Identifier(self.table_name),
                    col=sql.Identifier(self.case_config.array_field),
                ),
            )
            self.cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON public.{tbl} (pipeline_id)",
                ).format(
                    idx=sql.Identifier(f"{self.table_name}_pipeline_id_btree"),
                    tbl=sql.Identifier(self.table_name),
                ),
            )
        elif self.hybrid_mode == "join":
            doc = self.case_config.doc_table_name
            self.cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON public.{doc} USING gin ({col})",
                ).format(
                    idx=sql.Identifier(f"{doc}_{self.case_config.doc_tags_field}_gin"),
                    doc=sql.Identifier(doc),
                    col=sql.Identifier(self.case_config.doc_tags_field),
                ),
            )
            self.cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON public.{tbl} ({col})",
                ).format(
                    idx=sql.Identifier(f"{self.table_name}_{self.case_config.doc_join_field}_btree"),
                    tbl=sql.Identifier(self.table_name),
                    col=sql.Identifier(self.case_config.doc_join_field),
                ),
            )
        self.conn.commit()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        if self.with_scalar_labels:
            assert labels_data is not None, "labels_data should be provided if with_scalar_labels is set to True"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            base_types: list[Any] = ["bigint", "vector"]
            if self.with_scalar_labels:
                base_types.append("varchar")
            if self.hybrid_mode == "array":
                # 1009 = text[] (psycopg registry has no '_text' alias).
                base_types.extend([1009, "text"])
            elif self.hybrid_mode == "join":
                base_types.append("bigint")

            with self.cursor.copy(
                sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                    table_name=sql.Identifier(self.table_name),
                ),
            ) as copy:
                copy.set_types(base_types)
                for i, row in enumerate(metadata_arr):
                    rid = int(row)
                    values: list[Any] = [rid, embeddings_arr[i]]
                    if self.with_scalar_labels:
                        values.append(labels_data[i])
                    if self.hybrid_mode == "array":
                        values.append(hybrid_synth.user_array_for(rid))
                        values.append(hybrid_synth.pipeline_id_for(rid))
                    elif self.hybrid_mode == "join":
                        values.append(hybrid_synth.pipeline_doc_id_for(rid))
                    copy.write_row(tuple(values))
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into adbpg table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"WHERE {self._primary_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = f"WHERE {self._scalar_label_field} = '{filters.label_value}'"
        elif filters.type == FilterOp.ArrayContains:
            vals = ",".join(f"'{v}'" for v in filters.values)
            self.where_clause = f"WHERE {filters.array_field} @> ARRAY[{vals}]::text[]"
        elif filters.type == FilterOp.JoinArrayOverlap:
            vals = ",".join(f"'{v}'" for v in filters.values)
            self.where_clause = (
                f"WHERE {filters.join_field} IN (SELECT {filters.join_field} FROM public.{filters.doc_table} "
                f"WHERE {filters.tags_field} && ARRAY[{vals}]::text[])"
            )
        else:
            msg = f"Not support Filter for Adbpg - {filters}"
            raise ValueError(msg)

        self._search = self._generate_search_query()

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        q = np.asarray(query)

        if (
            getattr(self.case_config, "enable_explain_analyze", False)
            and self._explain_count < getattr(self.case_config, "explain_sample_size", 0)
        ):
            try:
                explain_sql = sql.SQL("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ") + self._search
                rows = self.cursor.execute(explain_sql, (q, k), prepare=False, binary=False).fetchall()
                # Plan rows come back as a single-element list of JSON.
                plan = rows[0][0] if rows else None
                self.explain_plans.append({"query_idx": self._explain_count, "plan": plan})
                self._persist_explain_plan(plan)
                self._explain_count += 1
            except Exception as e:  # pragma: no cover - diagnostic side-channel
                log.warning(f"EXPLAIN capture failed: {e}")
                self._explain_count += 1

        result = self.cursor.execute(
            self._search,
            (q, k),
            prepare=True,
            binary=True,
        )
        return [int(i[0]) for i in result.fetchall()]

    def _persist_explain_plan(self, plan: Any) -> None:
        """Append captured EXPLAIN ANALYZE JSON to a per-run jsonl file."""
        try:
            if not getattr(self, "_explain_log_path", None):
                d = "/tmp/vdb_explain"
                os.makedirs(d, exist_ok=True)
                self._explain_log_path = os.path.join(
                    d, f"{self.table_name}_{int(time.time())}.jsonl",
                )
            with open(self._explain_log_path, "a") as fh:
                fh.write(json.dumps({
                    "ts": time.time(),
                    "table": self.table_name,
                    "hybrid_mode": self.hybrid_mode,
                    "plan": plan,
                }) + "\n")
        except Exception as e:
            log.warning(f"persist EXPLAIN failed: {e}")
