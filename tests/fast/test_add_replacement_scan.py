import pytest

import duckdb

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


class TestAddReplacementScanDataFrame:
    """Tests for callbacks that return scannable Python objects."""

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


class TestAddReplacementScanTableRedirect:
    """Tests for callbacks that return {"type": "table", ...} name redirects."""

    def test_basic_redirect(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE real_tbl (a INT)")
        duckdb_cursor.execute("INSERT INTO real_tbl VALUES (1), (2)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "table", "table": "real_tbl"} if table_name == "tbl" else None
        )
        res = duckdb_cursor.sql("SELECT * FROM tbl ORDER BY a").fetchall()
        assert res == [(1,), (2,)]

    def test_redirect_with_alias(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE real_tbl (id INT, name TEXT)")
        duckdb_cursor.execute("INSERT INTO real_tbl VALUES (1, 'Widget')")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "table", "table": "real_tbl"} if table_name == "tbl" else None
        )
        res = duckdb_cursor.sql("SELECT t.name FROM tbl AS t WHERE t.id = 1").fetchall()
        assert res == [("Widget",)]

    def test_redirect_with_where(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE real_tbl (id INT, val DOUBLE)")
        duckdb_cursor.execute("INSERT INTO real_tbl VALUES (1, 10.0), (2, 20.0), (3, 30.0)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "table", "table": "real_tbl"} if table_name == "tbl" else None
        )
        res = duckdb_cursor.sql("SELECT id FROM tbl WHERE val > 15 ORDER BY id").fetchall()
        assert res == [(2,), (3,)]

    def test_redirect_self_join(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE real_tbl (id INT, name TEXT)")
        duckdb_cursor.execute("INSERT INTO real_tbl VALUES (1, 'A'), (2, 'B')")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "table", "table": "real_tbl"} if table_name == "tbl" else None
        )
        res = duckdb_cursor.sql(
            "SELECT a.name, b.name FROM tbl AS a JOIN tbl AS b ON a.id = b.id ORDER BY a.id"
        ).fetchall()
        assert res == [("A", "A"), ("B", "B")]

    def test_redirect_join_with_native(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE products (id INT, name TEXT)")
        duckdb_cursor.execute("INSERT INTO products VALUES (1, 'W'), (2, 'G')")
        duckdb_cursor.execute("CREATE TABLE prices (product_id INT, price DOUBLE)")
        duckdb_cursor.execute("INSERT INTO prices VALUES (1, 9.99), (2, 19.50)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "table", "table": "products"} if table_name == "items" else None
        )
        res = duckdb_cursor.sql(
            "SELECT i.name, pr.price FROM items AS i JOIN prices AS pr ON i.id = pr.product_id ORDER BY i.name"
        ).fetchall()
        assert res == [("G", 19.5), ("W", 9.99)]

    def test_redirect_with_schema(self, duckdb_cursor):
        """Redirect using schema_name from the callback."""
        duckdb_cursor.execute("CREATE TABLE real_tbl (v INT)")
        duckdb_cursor.execute("INSERT INTO real_tbl VALUES (42)")

        def resolver(table_name, schema_name="", **kw):
            if schema_name == "virtual":
                return {"type": "table", "table": "real_tbl"}
            return None

        duckdb_cursor.add_replacement_scan(resolver)
        res = duckdb_cursor.sql("SELECT v FROM virtual.anything").fetchall()
        assert res == [(42,)]

    def test_redirect_to_qualified_name(self, duckdb_cursor):
        """Redirect to a table in a specific schema."""
        duckdb_cursor.execute("CREATE SCHEMA other")
        duckdb_cursor.execute("CREATE TABLE other.data (x INT)")
        duckdb_cursor.execute("INSERT INTO other.data VALUES (99)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "table", "schema": "other", "table": "data"}
            if table_name == "tbl"
            else None
        )
        res = duckdb_cursor.sql("SELECT * FROM tbl").fetchall()
        assert res == [(99,)]

    def test_redirect_to_nonexistent(self, duckdb_cursor):
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "table", "table": "does_not_exist"}
            if table_name == "tbl"
            else None
        )
        with pytest.raises(duckdb.CatalogException, match="does not exist"):
            duckdb_cursor.sql("SELECT * FROM tbl").fetchall()

    def test_redirect_create_view(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE real_tbl (a INT)")
        duckdb_cursor.execute("INSERT INTO real_tbl VALUES (1), (2), (3)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "table", "table": "real_tbl"} if table_name == "tbl" else None
        )
        duckdb_cursor.execute("CREATE VIEW v AS SELECT a * 2 AS doubled FROM tbl")
        res = duckdb_cursor.sql("SELECT * FROM v ORDER BY doubled").fetchall()
        assert res == [(2,), (4,), (6,)]

    def test_redirect_ctas(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE real_tbl (a INT)")
        duckdb_cursor.execute("INSERT INTO real_tbl VALUES (10), (20)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "table", "table": "real_tbl"} if table_name == "tbl" else None
        )
        duckdb_cursor.execute("CREATE TABLE copy AS SELECT * FROM tbl")
        res = duckdb_cursor.sql("SELECT * FROM copy ORDER BY a").fetchall()
        assert res == [(10,), (20,)]


class TestAddReplacementScanQueryRewrite:
    """Tests for callbacks that return {"type": "query", "sql": ...} SQL rewrites."""

    def test_basic_rewrite(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE base (id INT, name TEXT, price DOUBLE)")
        duckdb_cursor.execute("INSERT INTO base VALUES (1, 'W', 9.99), (2, 'G', 19.50)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "query", "sql": "SELECT * FROM base WHERE price < 15"}
            if table_name == "cheap"
            else None
        )
        res = duckdb_cursor.sql("SELECT * FROM cheap").fetchall()
        assert res == [(1, "W", 9.99)]

    def test_rewrite_with_alias(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE base (id INT, name TEXT)")
        duckdb_cursor.execute("INSERT INTO base VALUES (1, 'A'), (2, 'B')")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "query", "sql": "SELECT * FROM base"}
            if table_name == "tbl"
            else None
        )
        res = duckdb_cursor.sql("SELECT t.name FROM tbl AS t WHERE t.id = 2").fetchall()
        assert res == [("B",)]

    def test_rewrite_filter_on_top(self, duckdb_cursor):
        """User's WHERE clause composes with the rewrite's WHERE clause."""
        duckdb_cursor.execute("CREATE TABLE base (id INT, category TEXT, price DOUBLE)")
        duckdb_cursor.execute("INSERT INTO base VALUES (1, 'A', 5), (2, 'A', 15), (3, 'B', 8)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "query", "sql": "SELECT * FROM base WHERE price < 10"}
            if table_name == "cheap"
            else None
        )
        res = duckdb_cursor.sql("SELECT id FROM cheap WHERE category = 'A'").fetchall()
        assert res == [(1,)]

    def test_rewrite_with_aggregation(self, duckdb_cursor):
        """Rewrite SQL contains aggregation."""
        duckdb_cursor.execute("CREATE TABLE base (category TEXT, val INT)")
        duckdb_cursor.execute("INSERT INTO base VALUES ('A', 1), ('A', 2), ('B', 3)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {
                "type": "query",
                "sql": "SELECT category, SUM(val) AS total FROM base GROUP BY category",
            }
            if table_name == "summary"
            else None
        )
        res = duckdb_cursor.sql("SELECT * FROM summary ORDER BY category").fetchall()
        assert res == [("A", 3), ("B", 3)]

    def test_rewrite_with_table_function(self, duckdb_cursor):
        """Rewrite SQL uses a table function (range)."""
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "query", "sql": "SELECT i AS id FROM range(5) t(i)"}
            if table_name == "seq"
            else None
        )
        res = duckdb_cursor.sql("SELECT * FROM seq ORDER BY id").fetchall()
        assert res == [(0,), (1,), (2,), (3,), (4,)]

    def test_rewrite_join_with_redirect(self, duckdb_cursor):
        """Join a SQL rewrite with a table redirect."""
        duckdb_cursor.execute("CREATE TABLE products (id INT, name TEXT, price DOUBLE)")
        duckdb_cursor.execute("INSERT INTO products VALUES (1, 'W', 9.99), (2, 'G', 19.50), (3, 'D', 4.99)")

        def resolver(table_name, **kw):
            if table_name == "all_products":
                return {"type": "table", "table": "products"}
            if table_name == "cheap":
                return {"type": "query", "sql": "SELECT * FROM products WHERE price < 10"}
            return None

        duckdb_cursor.add_replacement_scan(resolver)
        res = duckdb_cursor.sql(
            "SELECT p.name, c.price FROM all_products AS p JOIN cheap AS c ON p.id = c.id ORDER BY p.name"
        ).fetchall()
        assert res == [("D", 4.99), ("W", 9.99)]

    def test_rewrite_self_join(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE base (id INT, name TEXT)")
        duckdb_cursor.execute("INSERT INTO base VALUES (1, 'A'), (2, 'B')")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "query", "sql": "SELECT * FROM base"}
            if table_name == "tbl"
            else None
        )
        res = duckdb_cursor.sql(
            "SELECT a.name, b.name FROM tbl AS a JOIN tbl AS b ON a.id = b.id ORDER BY a.id"
        ).fetchall()
        assert res == [("A", "A"), ("B", "B")]

    def test_rewrite_cte(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE base (id INT, val INT)")
        duckdb_cursor.execute("INSERT INTO base VALUES (1, 10), (2, 20), (3, 30)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "query", "sql": "SELECT * FROM base WHERE val > 10"}
            if table_name == "filtered"
            else None
        )
        res = duckdb_cursor.sql(
            "WITH top AS (SELECT * FROM filtered ORDER BY val DESC LIMIT 1) SELECT * FROM top"
        ).fetchall()
        assert res == [(3, 30)]

    def test_rewrite_subquery(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE base (id INT, val INT)")
        duckdb_cursor.execute("INSERT INTO base VALUES (1, 10), (2, 20), (3, 30)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "query", "sql": "SELECT * FROM base"}
            if table_name == "tbl"
            else None
        )
        res = duckdb_cursor.sql(
            "SELECT * FROM tbl WHERE val > (SELECT AVG(val) FROM tbl) ORDER BY id"
        ).fetchall()
        assert res == [(3, 30)]

    def test_rewrite_window(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE base (id INT, val INT)")
        duckdb_cursor.execute("INSERT INTO base VALUES (1, 30), (2, 10), (3, 20)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "query", "sql": "SELECT * FROM base"}
            if table_name == "tbl"
            else None
        )
        res = duckdb_cursor.sql(
            "SELECT id, RANK() OVER (ORDER BY val) AS rnk FROM tbl ORDER BY rnk"
        ).fetchall()
        assert res == [(2, 1), (3, 2), (1, 3)]

    def test_rewrite_prepared(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE base (id INT, name TEXT)")
        duckdb_cursor.execute("INSERT INTO base VALUES (1, 'A'), (2, 'B')")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "query", "sql": "SELECT * FROM base"}
            if table_name == "tbl"
            else None
        )
        res = duckdb_cursor.execute("SELECT name FROM tbl WHERE id = ?", [2]).fetchall()
        assert res == [("B",)]

    def test_rewrite_create_view(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE base (a INT)")
        duckdb_cursor.execute("INSERT INTO base VALUES (1), (2), (3)")
        duckdb_cursor.add_replacement_scan(
            lambda table_name, **kw: {"type": "query", "sql": "SELECT a * 10 AS x FROM base"}
            if table_name == "tbl"
            else None
        )
        duckdb_cursor.execute("CREATE VIEW v AS SELECT x FROM tbl WHERE x > 10")
        res = duckdb_cursor.sql("SELECT * FROM v ORDER BY x").fetchall()
        assert res == [(20,), (30,)]

    def test_invalid_sql(self, duckdb_cursor):
        duckdb_cursor.add_replacement_scan(
            lambda **kw: {"type": "query", "sql": "NOT VALID SQL !!!"}
        )
        with pytest.raises(Exception):
            duckdb_cursor.sql("SELECT * FROM tbl").fetchall()

    def test_non_select(self, duckdb_cursor):
        duckdb_cursor.add_replacement_scan(
            lambda **kw: {"type": "query", "sql": "CREATE TABLE x (a INT)"}
        )
        with pytest.raises(duckdb.InvalidInputException, match="SELECT"):
            duckdb_cursor.sql("SELECT * FROM tbl").fetchall()

    def test_missing_sql_key(self, duckdb_cursor):
        duckdb_cursor.add_replacement_scan(lambda **kw: {"type": "query"})
        with pytest.raises(duckdb.InvalidInputException, match="sql"):
            duckdb_cursor.sql("SELECT * FROM tbl").fetchall()


class TestAddReplacementScanDictErrors:
    """Tests for error handling in dict-based directives."""

    def test_dict_without_type(self, duckdb_cursor):
        duckdb_cursor.add_replacement_scan(lambda **kw: {"foo": "bar"})
        with pytest.raises(duckdb.InvalidInputException, match="type"):
            duckdb_cursor.sql("SELECT * FROM tbl").fetchall()

    def test_unknown_type(self, duckdb_cursor):
        duckdb_cursor.add_replacement_scan(lambda **kw: {"type": "magic"})
        with pytest.raises(duckdb.InvalidInputException, match="magic"):
            duckdb_cursor.sql("SELECT * FROM tbl").fetchall()


class TestAddReplacementScanMixed:
    """Tests for resolvers that return different types for different names."""

    def test_mixed_redirect_rewrite_dataframe(self, duckdb_cursor):
        duckdb_cursor.execute("CREATE TABLE real_tbl (v INT)")
        duckdb_cursor.execute("INSERT INTO real_tbl VALUES (1), (2)")

        def resolver(table_name, **kw):
            if table_name == "via_redirect":
                return {"type": "table", "table": "real_tbl"}
            if table_name == "via_query":
                return {"type": "query", "sql": "SELECT v * 10 AS v FROM real_tbl"}
            if table_name == "via_df":
                return pd.DataFrame({"v": [100, 200]})
            if table_name == "via_arrow":
                return pa.table({"v": [1000, 2000]})
            return None

        duckdb_cursor.add_replacement_scan(resolver)
        r1 = duckdb_cursor.sql("SELECT SUM(v) FROM via_redirect").fetchone()[0]
        r2 = duckdb_cursor.sql("SELECT SUM(v) FROM via_query").fetchone()[0]
        r3 = duckdb_cursor.sql("SELECT SUM(v) FROM via_df").fetchone()[0]
        r4 = duckdb_cursor.sql("SELECT SUM(v) FROM via_arrow").fetchone()[0]
        assert r1 == 3
        assert r2 == 30
        assert r3 == 300
        assert r4 == 3000
