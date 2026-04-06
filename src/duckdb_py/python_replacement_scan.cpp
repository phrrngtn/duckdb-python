#include "duckdb_python/python_replacement_scan.hpp"
#include "duckdb/main/db_instance_cache.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/main/client_properties.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pybind11/dataframe.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb_python/pandas/pandas_scan.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb_python/pyrelation.hpp"
#include <duckdb/main/settings.hpp>

using namespace pybind11::literals; // for _a keyword argument syntax

namespace duckdb {

static void CreateArrowScan(const string &name, py::object entry, TableFunctionRef &table_function,
                            vector<unique_ptr<ParsedExpression>> &children, ClientProperties &client_properties,
                            PyArrowObjectType type, DatabaseInstance &db) {
	shared_ptr<ExternalDependency> external_dependency = make_shared_ptr<ExternalDependency>();
	if (type == PyArrowObjectType::MessageReader) {
		if (!db.ExtensionIsLoaded("nanoarrow")) {
			throw MissingExtensionException(
			    "The nanoarrow community extension is needed to read the Arrow IPC protocol. \n You can install it "
			    "with \"INSTALL nanoarrow FROM community;\". \n Then you can load it with \"LOAD nanoarrow;\"");
		}
		vector<Value> values;
		py::list stream_messages;
		while (true) {
			try {
				py::object message = entry.attr("read_next_message")();
				if (message.is_none()) {
					break;
				}
				stream_messages.append(message.attr("serialize")());
				const auto buffer_address = stream_messages[stream_messages.size() - 1].attr("address").cast<int64_t>();
				const auto buffer_size = stream_messages[stream_messages.size() - 1].attr("size").cast<uint32_t>();
				child_list_t<Value> buffer_values;
				buffer_values.push_back({"ptr", Value::POINTER(buffer_address)});
				buffer_values.push_back({"size", Value::UBIGINT(buffer_size)});
				values.push_back(Value::STRUCT(buffer_values));
			} catch (const py::error_already_set &e) {
				break;
			}
		}
		auto list_value = Value::LIST(values);
		children.push_back(make_uniq<ConstantExpression>(list_value));
		table_function.function = make_uniq<FunctionExpression>("scan_arrow_ipc", std::move(children));
		auto dependency_item = PythonDependencyItem::Create(stream_messages);
		external_dependency->AddDependency("replacement_cache", std::move(dependency_item));
	} else {
		auto stream_factory = make_uniq<PythonTableArrowArrayStreamFactory>(entry.ptr(), client_properties, type);
		auto stream_factory_produce = PythonTableArrowArrayStreamFactory::Produce;
		auto stream_factory_get_schema = PythonTableArrowArrayStreamFactory::GetSchema;

		children.push_back(make_uniq<ConstantExpression>(Value::POINTER(CastPointerToValue(stream_factory.get()))));
		children.push_back(make_uniq<ConstantExpression>(Value::POINTER(CastPointerToValue(stream_factory_produce))));
		children.push_back(
		    make_uniq<ConstantExpression>(Value::POINTER(CastPointerToValue(stream_factory_get_schema))));

		if (type == PyArrowObjectType::PyCapsule) {
			// Disable projection+filter pushdown for bare capsules (single-use, no PyArrow wrapper)
			table_function.function = make_uniq<FunctionExpression>("arrow_scan_dumb", std::move(children));
		} else if (type == PyArrowObjectType::PyCapsuleInterface) {
			// Try to load pyarrow.dataset for pushdown support
			auto &cache = *DuckDBPyConnection::ImportCache();
			if (!cache.pyarrow.dataset()) {
				// No pyarrow.dataset: scan without pushdown, DuckDB handles projection/filter post-scan
				table_function.function = make_uniq<FunctionExpression>("arrow_scan_dumb", std::move(children));
			} else {
				table_function.function = make_uniq<FunctionExpression>("arrow_scan", std::move(children));
			}
		} else {
			table_function.function = make_uniq<FunctionExpression>("arrow_scan", std::move(children));
		}
		auto dependency_item =
		    PythonDependencyItem::Create(make_uniq<RegisteredArrow>(std::move(stream_factory), entry));
		external_dependency->AddDependency("replacement_cache", std::move(dependency_item));
	}
	table_function.external_dependency = std::move(external_dependency);
}

static void ThrowScanFailureError(const py::object &entry, const string &name, const string &location = "") {
	string error;
	auto py_object_type = string(py::str(py::type::of(entry).attr("__name__")));
	error += StringUtil::Format("Python Object \"%s\" of type \"%s\"", name, py_object_type);
	if (!location.empty()) {
		error += StringUtil::Format(" found on line \"%s\"", location);
	}
	error +=
	    StringUtil::Format(" not suitable for replacement scans.\nMake sure "
	                       "that \"%s\" is either a pandas.DataFrame, duckdb.DuckDBPyRelation, pyarrow Table, Dataset, "
	                       "RecordBatchReader, Scanner, or NumPy ndarrays with supported format",
	                       name);
	throw InvalidInputException(error);
}

unique_ptr<TableRef> PythonReplacementScan::ReplacementObject(const py::object &entry, const string &name,
                                                              ClientContext &context, bool relation) {
	auto replacement = TryReplacementObject(entry, name, context, relation);
	if (!replacement) {
		ThrowScanFailureError(entry, name);
	}
	return replacement;
}

unique_ptr<TableRef> PythonReplacementScan::TryReplacementObject(const py::object &entry, const string &name,
                                                                 ClientContext &context, bool relation) {
	auto client_properties = context.GetClientProperties();
	auto table_function = make_uniq<TableFunctionRef>();
	vector<unique_ptr<ParsedExpression>> children;
	NumpyObjectType numpytype;
	PyArrowObjectType arrow_type;
	if (DuckDBPyConnection::IsPandasDataframe(entry)) {
		if (PandasDataFrame::IsPyArrowBacked(entry)) {
			auto table = PandasDataFrame::ToArrowTable(entry);
			CreateArrowScan(name, table, *table_function, children, client_properties, PyArrowObjectType::Table,
			                *context.db);
		} else {
			string name = "df_" + StringUtil::GenerateRandomName();
			auto new_df = PandasScanFunction::PandasReplaceCopiedNames(entry);
			children.push_back(make_uniq<ConstantExpression>(Value::POINTER(CastPointerToValue(new_df.ptr()))));
			table_function->function = make_uniq<FunctionExpression>("pandas_scan", std::move(children));
			auto dependency = make_uniq<ExternalDependency>();
			dependency->AddDependency("replacement_cache", PythonDependencyItem::Create(entry));
			dependency->AddDependency("copy", PythonDependencyItem::Create(new_df));
			table_function->external_dependency = std::move(dependency);
		}
	} else if (DuckDBPyRelation::IsRelation(entry)) {
		auto pyrel = py::cast<DuckDBPyRelation *>(entry);
		if (!pyrel->CanBeRegisteredBy(context)) {
			throw InvalidInputException(
			    "Python Object \"%s\" of type \"DuckDBPyRelation\" not suitable for replacement scan.\nThe object was "
			    "created by another Connection and can therefore not be used by this Connection.",
			    name);
		}
		// create a subquery from the underlying relation object
		auto select = make_uniq<SelectStatement>();
		select->node = pyrel->GetRel().GetQueryNode();
		auto subquery = make_uniq<SubqueryRef>(std::move(select));
		auto dependency = make_uniq<ExternalDependency>();
		dependency->AddDependency("replacement_cache", PythonDependencyItem::Create(entry));
		subquery->external_dependency = std::move(dependency);
		return std::move(subquery);
	} else if (PolarsDataFrame::IsDataFrame(entry)) {
		// Polars DataFrames always go through one-time .to_arrow() materialization.
		// Polars's __arrow_c_stream__() serializes from its internal layout on every call,
		// which is expensive for repeated scans. The .to_arrow() path converts once.
		auto arrow_dataset = entry.attr("to_arrow")();
		CreateArrowScan(name, arrow_dataset, *table_function, children, client_properties, PyArrowObjectType::Table,
		                *context.db);
	} else if (PolarsDataFrame::IsLazyFrame(entry)) {
		CreateArrowScan(name, entry, *table_function, children, client_properties, PyArrowObjectType::PolarsLazyFrame,
		                *context.db);
	} else if ((arrow_type = DuckDBPyConnection::GetArrowType(entry)) != PyArrowObjectType::Invalid &&
	           !(arrow_type == PyArrowObjectType::MessageReader && !relation)) {
		CreateArrowScan(name, entry, *table_function, children, client_properties, arrow_type, *context.db);
	} else if (DuckDBPyConnection::IsAcceptedNumpyObject(entry) != NumpyObjectType::INVALID) {
		numpytype = DuckDBPyConnection::IsAcceptedNumpyObject(entry);
		string np_name = "np_" + StringUtil::GenerateRandomName();
		py::dict data; // we will convert all the supported format to dict{"key": np.array(value)}.
		size_t idx = 0;
		switch (numpytype) {
		case NumpyObjectType::NDARRAY1D:
			data["column0"] = entry;
			break;
		case NumpyObjectType::NDARRAY2D:
			idx = 0;
			for (auto item : py::cast<py::array>(entry)) {
				data[("column" + std::to_string(idx)).c_str()] = item;
				idx++;
			}
			break;
		case NumpyObjectType::LIST:
			idx = 0;
			for (auto item : py::cast<py::list>(entry)) {
				data[("column" + std::to_string(idx)).c_str()] = item;
				idx++;
			}
			break;
		case NumpyObjectType::DICT:
			data = py::cast<py::dict>(entry);
			break;
		default:
			throw NotImplementedException("Unsupported Numpy object");
			break;
		}
		children.push_back(make_uniq<ConstantExpression>(Value::POINTER(CastPointerToValue(data.ptr()))));
		table_function->function = make_uniq<FunctionExpression>("pandas_scan", std::move(children));
		auto dependency = make_uniq<ExternalDependency>();
		dependency->AddDependency("replacement_cache", PythonDependencyItem::Create(entry));
		dependency->AddDependency("data", PythonDependencyItem::Create(data));
		table_function->external_dependency = std::move(dependency);
	} else {
		// This throws an error later on!
		return nullptr;
	}
	return std::move(table_function);
}

static bool IsBuiltinFunction(const py::object &object) {
	auto &import_cache_py = *DuckDBPyConnection::ImportCache();
	return py::isinstance(object, import_cache_py.types.BuiltinFunctionType());
}

static unique_ptr<TableRef> TryReplacement(py::dict &dict, const string &name, ClientContext &context,
                                           py::object &current_frame) {
	auto table_name = py::str(name);
	if (!dict.contains(table_name)) {
		// not present in the globals
		return nullptr;
	}
	const py::object &entry = dict[table_name];

	if (IsBuiltinFunction(entry)) {
		return nullptr;
	}

	auto result = PythonReplacementScan::TryReplacementObject(entry, name, context);
	if (!result) {
		std::string location = py::cast<py::str>(current_frame.attr("f_code").attr("co_filename"));
		location += ":";
		location += py::cast<py::str>(current_frame.attr("f_lineno"));
		ThrowScanFailureError(entry, name, location);
	}
	return result;
}

static unique_ptr<TableRef> ReplaceInternal(ClientContext &context, const string &table_name) {
	Value result;
	auto lookup_result = context.TryGetCurrentSetting("python_enable_replacements", result);
	D_ASSERT((bool)lookup_result);
	auto enabled = result.GetValue<bool>();

	if (!enabled) {
		return nullptr;
	}

	lookup_result = context.TryGetCurrentSetting("python_scan_all_frames", result);
	D_ASSERT((bool)lookup_result);
	auto scan_all_frames = result.GetValue<bool>();

	py::gil_scoped_acquire acquire;
	py::object current_frame;
	try {
		current_frame = py::module::import("inspect").attr("currentframe")();
	} catch (py::error_already_set &e) {
		//! Likely no call stack exists, just safely return
		return nullptr;
	}

	bool has_locals = false;
	bool has_globals = false;
	do {
		if (py::none().is(current_frame)) {
			break;
		}

		py::object local_dict_p;
		try {
			local_dict_p = current_frame.attr("f_locals");
		} catch (py::error_already_set &e) {
			return nullptr;
		}
		has_locals = !py::none().is(local_dict_p);
		if (has_locals) {
			// search local dictionary
			auto local_dict = py::cast<py::dict>(local_dict_p);
			auto result = TryReplacement(local_dict, table_name, context, current_frame);
			if (result) {
				return result;
			}
		}
		py::object global_dict_p;
		try {
			global_dict_p = current_frame.attr("f_globals");
		} catch (py::error_already_set &e) {
			return nullptr;
		}
		has_globals = !py::none().is(global_dict_p);
		if (has_globals) {
			auto global_dict = py::cast<py::dict>(global_dict_p);
			// search global dictionary
			auto result = TryReplacement(global_dict, table_name, context, current_frame);
			if (result) {
				return result;
			}
		}
		try {
			current_frame = current_frame.attr("f_back");
		} catch (py::error_already_set &e) {
			return nullptr;
		}
	} while (scan_all_frames && (has_locals || has_globals));
	return nullptr;
}

unique_ptr<TableRef> PythonReplacementScan::Replace(ClientContext &context, ReplacementScanInput &input,
                                                    optional_ptr<ReplacementScanData> data) {
	auto &table_name = input.table_name;
	auto &config = DBConfig::GetConfig(context);
	if (!Settings::Get<EnableExternalAccessSetting>(config)) {
		return nullptr;
	}

	unique_ptr<TableRef> result;
	result = ReplaceInternal(context, table_name);
	return result;
}

// Handle a dict-based replacement scan directive from the Python callback.
//
// The dict must contain a "type" key that determines the replacement strategy:
//
// {"type": "table", "catalog": "...", "schema": "...", "table": "..."}
//   Redirects the unresolved name to an existing catalog entry. Constructs a
//   BaseTableRef wrapped in a SubqueryRef (the wrapping is necessary because the
//   binder's replacement scan code overwrites the alias on non-SubqueryRef results,
//   which would break user-provided aliases like "FROM products AS p").
//   The binder resolves the redirected name through normal catalog lookup, so full
//   optimizer pushdown applies — predicates, projections, partition pruning, etc.
//
// {"type": "query", "sql": "SELECT ..."}
//   Rewrites the unresolved name as an arbitrary SQL subquery. The SQL is parsed
//   using DuckDB's Parser directly (not through ClientContext::ParseStatements),
//   which avoids acquiring the ClientContext lock — critical because the binder
//   already holds that lock when the replacement scan fires. The parsed AST is
//   wrapped in a SubqueryRef and the optimizer sees through it.
//
// Both paths produce pure AST nodes with no connection dependencies and no data
// copying. The binder on the active connection resolves all table references in
// the constructed AST against its own catalog.
static unique_ptr<TableRef> HandleDictDirective(const py::dict &directive, const string &table_name,
                                                ClientContext &context) {
	if (!directive.contains("type")) {
		throw InvalidInputException("Replacement scan directive dict must contain a 'type' key");
	}
	auto type_str = py::str(directive["type"]).cast<string>();

	if (type_str == "table") {
		auto base_ref = make_uniq<BaseTableRef>();
		if (directive.contains("catalog")) {
			base_ref->catalog_name = py::str(directive["catalog"]).cast<string>();
		}
		if (directive.contains("schema")) {
			base_ref->schema_name = py::str(directive["schema"]).cast<string>();
		}
		if (directive.contains("table")) {
			base_ref->table_name = py::str(directive["table"]).cast<string>();
		} else {
			base_ref->table_name = table_name;
		}

		auto select_node = make_uniq<SelectNode>();
		select_node->select_list.push_back(make_uniq<StarExpression>());
		select_node->from_table = std::move(base_ref);
		auto select_stmt = make_uniq<SelectStatement>();
		select_stmt->node = std::move(select_node);
		auto subquery = make_uniq<SubqueryRef>(std::move(select_stmt));
		subquery->alias = table_name;
		return std::move(subquery);
	}

	if (type_str == "query") {
		// SQL rewrite: {"type": "query", "sql": "SELECT ..."}
		if (!directive.contains("sql")) {
			throw InvalidInputException("Replacement scan directive with type 'query' must contain a 'sql' key");
		}
		auto sql = py::str(directive["sql"]).cast<string>();

		// Parse the SQL directly — no lock needed, Parser is stateless.
		ParserOptions options;
		options.preserve_identifier_case = true;
		Parser parser(options);
		parser.ParseQuery(sql);

		if (parser.statements.empty()) {
			throw InvalidInputException("Replacement scan SQL produced no statements");
		}
		if (parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
			throw InvalidInputException("Replacement scan SQL must be a SELECT statement");
		}

		auto select_stmt = unique_ptr_cast<SQLStatement, SelectStatement>(std::move(parser.statements[0]));
		auto subquery = make_uniq<SubqueryRef>(std::move(select_stmt));
		subquery->alias = table_name;
		return std::move(subquery);
	}

	throw InvalidInputException("Unknown replacement scan directive type: '%s' (expected 'table' or 'query')",
	                            type_str);
}

// Entry point for user-registered replacement scans. Called by DuckDB's binder
// when it encounters an unresolved table name. The binder holds the ClientContext
// lock at this point, so the callback must not call back into the same connection
// (con.sql(), con.execute(), etc.) — that would deadlock.
//
// The GIL is acquired before calling into Python. If the callback raises a Python
// exception, we catch it and return nullptr (decline), letting DuckDB try the next
// replacement scan in the chain.
unique_ptr<TableRef> PythonCallbackReplacementScan(ClientContext &context, ReplacementScanInput &input,
                                                   optional_ptr<ReplacementScanData> data) {
	auto &config = DBConfig::GetConfig(context);
	if (!Settings::Get<EnableExternalAccessSetting>(config)) {
		return nullptr;
	}
	if (!data) {
		return nullptr;
	}
	auto &scan_data = data->Cast<PythonCallbackReplacementScanData>();

	py::gil_scoped_acquire acquire;
	py::object result;
	try {
		result = scan_data.callback("table_name"_a = py::str(input.table_name),
		                            "schema_name"_a = py::str(input.schema_name),
		                            "catalog_name"_a = py::str(input.catalog_name));
	} catch (py::error_already_set &e) {
		return nullptr;
	}

	// Callback returned None: decline, let the next replacement scan try.
	if (result.is_none()) {
		return nullptr;
	}

	// Callback returned a dict: structured directive (name redirect or SQL rewrite).
	// These paths construct AST nodes without any connection interaction.
	if (py::isinstance<py::dict>(result)) {
		return HandleDictDirective(result.cast<py::dict>(), input.table_name, context);
	}

	// Callback returned a Python object (DataFrame, Arrow Table, etc.):
	// use the existing machinery to convert it into a scannable TableRef.
	return PythonReplacementScan::TryReplacementObject(result, input.table_name, context);
}

} // namespace duckdb
