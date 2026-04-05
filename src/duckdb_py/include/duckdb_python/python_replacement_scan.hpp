#pragma once

#include "duckdb/main/client_context_state.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

//! ReplacementScanData subclass that holds a Python callable.
//! The callable signature is: callback(table_name: str) -> Optional[scannable]
//! where scannable is a pandas DataFrame, pyarrow Table/Dataset/RecordBatchReader,
//! DuckDBPyRelation, or any object supporting __arrow_c_stream__.
struct PythonCallbackReplacementScanData : public ReplacementScanData {
	explicit PythonCallbackReplacementScanData(py::function callback_p)
	    : callback(std::move(callback_p)) {}
	~PythonCallbackReplacementScanData() override {
		py::gil_scoped_acquire acquire;
		callback = py::function();
	}
	py::function callback;
};

//! Replacement scan function that delegates to a user-registered Python callback.
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
