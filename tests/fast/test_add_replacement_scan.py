import pytest

import duckdb

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


class TestAddReplacementScan:
    def test_basic(self, duckdb_cursor):
        def resolver(table_name, schema_name="", catalog_name=""):
            if table_name == "tbl":
                return pd.DataFrame({"a": [1, 2, 3]})
            return None

        duckdb_cursor.add_replacement_scan(resolver)
        res = duckdb_cursor.sql("SELECT * FROM tbl").fetchall()
        assert res == [(1,), (2,), (3,)]

    def test_returns_none(self, duckdb_cursor):
        duckdb_cursor.add_replacement_scan(lambda **kwargs: None)
        with pytest.raises(duckdb.CatalogException, match="Table with name tbl does not exist"):
            duckdb_cursor.sql("SELECT * FROM tbl").fetchall()

    def test_catalog_takes_priority(self, duckdb_cursor):
        call_log = []

        def resolver(table_name, **kwargs):
            call_log.append(table_name)
            return pd.DataFrame({"a": [999]})

        duckdb_cursor.add_replacement_scan(resolver)
        duckdb_cursor.execute("CREATE TABLE tbl (a INTEGER)")
        duckdb_cursor.execute("INSERT INTO tbl VALUES (1)")
        res = duckdb_cursor.sql("SELECT * FROM tbl").fetchall()
        assert res == [(1,)]
        assert "tbl" not in call_log

    def test_join_two_resolved(self, duckdb_cursor):
        tables = {
            "t1": pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}),
            "t2": pd.DataFrame({"id": [1, 2], "w": [10, 20]}),
        }
        duckdb_cursor.add_replacement_scan(lambda table_name, **kwargs: tables.get(table_name))
        res = duckdb_cursor.sql("SELECT t1.v, t2.w FROM t1 JOIN t2 USING (id) ORDER BY t1.v").fetchall()
        assert res == [("a", 10), ("b", 20)]

    def test_mixed_resolved_and_native(self, duckdb_cursor):
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kwargs: pd.DataFrame({"id": [1, 2], "v": ["a", "b"]})
            if table_name == "ext"
            else None
        )
        duckdb_cursor.execute("CREATE TABLE native (id INTEGER, w INTEGER)")
        duckdb_cursor.execute("INSERT INTO native VALUES (1, 10), (2, 20)")
        res = duckdb_cursor.sql("SELECT ext.v, native.w FROM ext JOIN native USING (id) ORDER BY ext.v").fetchall()
        assert res == [("a", 10), ("b", 20)]

    def test_arrow_table(self, duckdb_cursor):
        def resolver(table_name, **kwargs):
            if table_name == "tbl":
                return pa.table({"x": [4, 5, 6]})
            return None

        duckdb_cursor.add_replacement_scan(resolver)
        res = duckdb_cursor.sql("SELECT * FROM tbl").fetchall()
        assert res == [(4,), (5,), (6,)]

    def test_called_per_query(self, duckdb_cursor):
        call_count = 0

        def resolver(table_name, **kwargs):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame({"a": [call_count]})

        duckdb_cursor.add_replacement_scan(resolver)
        duckdb_cursor.sql("SELECT * FROM tbl").fetchall()
        duckdb_cursor.sql("SELECT * FROM tbl").fetchall()
        assert call_count >= 2

    def test_exception_in_callback(self, duckdb_cursor):
        def resolver(**kwargs):
            raise ValueError("boom")

        duckdb_cursor.add_replacement_scan(resolver)
        with pytest.raises(duckdb.CatalogException, match="Table with name tbl does not exist"):
            duckdb_cursor.sql("SELECT * FROM tbl").fetchall()

    def test_quoted_identifier(self, duckdb_cursor):
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kwargs: pd.DataFrame({"a": [1]}) if table_name == "Sheet1!A1:B10" else None
        )
        res = duckdb_cursor.sql('SELECT * FROM "Sheet1!A1:B10"').fetchall()
        assert res == [(1,)]

    def test_respects_enable_external_access(self):
        con = duckdb.connect(config={"enable_external_access": False})
        con.add_replacement_scan(lambda **kwargs: pd.DataFrame({"a": [1]}))
        with pytest.raises(duckdb.CatalogException, match="Table with name tbl does not exist"):
            con.sql("SELECT * FROM tbl").fetchall()

    def test_chaining(self, duckdb_cursor):
        result = duckdb_cursor.add_replacement_scan(lambda **kwargs: None)
        assert result is duckdb_cursor

    def test_schema_qualified(self, duckdb_cursor):
        """Two-part name (schema.table) passes schema_name to callback."""
        received = {}

        def resolver(table_name, schema_name="", catalog_name=""):
            received["table"] = table_name
            received["schema"] = schema_name
            received["catalog"] = catalog_name
            return pd.DataFrame({"a": [1]})

        duckdb_cursor.add_replacement_scan(resolver)
        duckdb_cursor.sql("SELECT * FROM Sheet1.products").fetchall()
        assert received["table"] == "products"
        assert received["schema"] == "Sheet1"
        assert received["catalog"] == ""

    def test_three_part_qualified(self, duckdb_cursor):
        """Three-part name (catalog.schema.table) passes all components."""
        received = {}

        def resolver(table_name, schema_name="", catalog_name=""):
            received["table"] = table_name
            received["schema"] = schema_name
            received["catalog"] = catalog_name
            return pd.DataFrame({"a": [1]})

        duckdb_cursor.add_replacement_scan(resolver)
        duckdb_cursor.sql('SELECT * FROM "Workbook"."Sheet1"."products"').fetchall()
        assert received["table"] == "products"
        assert received["schema"] == "Sheet1"
        assert received["catalog"] == "Workbook"
