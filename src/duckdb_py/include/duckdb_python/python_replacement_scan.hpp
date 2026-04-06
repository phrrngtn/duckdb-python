#pragma once

#include "duckdb/main/client_context_state.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

//! ReplacementScanData subclass that holds a user-registered Python callable.
//!
//! The callable is invoked with keyword arguments (table_name, schema_name, catalog_name)
//! when DuckDB encounters an unresolved table name during query planning. It can return:
//!   - None: decline (DuckDB tries the next replacement scan)
//!   - dict {"type": "table", "catalog": ..., "schema": ..., "table": ...}:
//!       name redirect — resolves to another catalog entry with full optimizer pushdown
//!   - dict {"type": "query", "sql": "SELECT ..."}:
//!       SQL rewrite — parsed lock-free into a SubqueryRef with full pushdown
//!   - a scannable Python object (DataFrame, Arrow Table, etc.):
//!       scanned directly via TryReplacementObject
//!
//! The dict-dispatch paths ("table" and "query") construct AST nodes without acquiring
//! any connection lock, which avoids the deadlock that would occur if the callback
//! tried to call con.sql() or con.execute() on the same connection (the binder holds
//! the ClientContext lock during replacement scan resolution).
struct PythonCallbackReplacementScanData : public ReplacementScanData {
	explicit PythonCallbackReplacementScanData(py::function callback_p)
	    : callback(std::move(callback_p)) {}

	//! The destructor must acquire the GIL before releasing the py::function.
	//! py::function wraps a PyObject* with RAII reference counting — its destructor
	//! calls Py_DECREF, which is a Python C API call that requires the GIL.
	//! This destructor runs when config.replacement_scans is cleared during database
	//! shutdown, on a C++ thread that does not hold the GIL. Without the acquire,
	//! the Py_DECREF would corrupt the interpreter state (observed as SIGABRT).
	~PythonCallbackReplacementScanData() override {
		py::gil_scoped_acquire acquire;
		callback = py::function();
	}

	py::function callback;
};

//! Replacement scan function that delegates to a user-registered Python callback.
//! Acquires the GIL, calls the callback, and dispatches on the return type.
unique_ptr<TableRef> PythonCallbackReplacementScan(ClientContext &context, ReplacementScanInput &input,
                                                   optional_ptr<ReplacementScanData> data);

struct PythonReplacementScan {
public:
	static unique_ptr<TableRef> Replace(ClientContext &context, ReplacementScanInput &input,
	                                    optional_ptr<ReplacementScanData> data);
	//! Try to perform a replacement, returns NULL on error
	static unique_ptr<TableRef> TryReplacementObject(const py::object &entry, const string &name,
	                                                 ClientContext &context, bool relation = false);
	//! Perform a replacement or throw if it failed
	static unique_ptr<TableRef> ReplacementObject(const py::object &entry, const string &name, ClientContext &context,
	                                              bool relation = false);
};

} // namespace duckdb
