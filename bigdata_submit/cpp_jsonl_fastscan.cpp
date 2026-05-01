#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <fstream>
#include <string>
#include <cctype>

static bool extract_json_string_value(const std::string &line, const std::string &key, std::string &out) {
    const std::string pattern = "\"" + key + "\":";
    size_t pos = line.find(pattern);
    if (pos == std::string::npos) return false;
    pos += pattern.size();
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
    if (pos >= line.size() || line[pos] != '"') return false;
    ++pos;
    std::string value;
    bool escape = false;
    for (; pos < line.size(); ++pos) {
        char c = line[pos];
        if (escape) {
            value.push_back(c);
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"') {
            out = value;
            return true;
        }
        value.push_back(c);
    }
    return false;
}

static bool extract_json_int_value(const std::string &line, const std::string &key, long long &out) {
    const std::string pattern = "\"" + key + "\":";
    size_t pos = line.find(pattern);
    if (pos == std::string::npos) return false;
    pos += pattern.size();
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
    if (pos >= line.size()) return false;
    bool negative = false;
    if (line[pos] == '-') {
        negative = true;
        ++pos;
    }
    if (pos >= line.size() || !std::isdigit(static_cast<unsigned char>(line[pos]))) return false;
    long long value = 0;
    while (pos < line.size() && std::isdigit(static_cast<unsigned char>(line[pos]))) {
        value = value * 10 + (line[pos] - '0');
        ++pos;
    }
    out = negative ? -value : value;
    return true;
}

static PyObject* scan_jsonl_stats(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* path = nullptr;
    int threshold = 1000;
    static const char* kwlist[] = {"path", "threshold", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|i", const_cast<char**>(kwlist), &path, &threshold)) {
        return nullptr;
    }

    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input.is_open()) {
        PyErr_SetString(PyExc_FileNotFoundError, "Cannot open JSONL file");
        return nullptr;
    }

    long long rows = 0;
    long long matched_rows = 0;
    long long sum_text_len = 0;
    std::string line;
    std::string domain;
    long long text_len = 0;

    while (std::getline(input, line)) {
        ++rows;
        domain.clear();
        text_len = 0;
        bool has_domain = extract_json_string_value(line, "domain", domain);
        bool has_text_len = extract_json_int_value(line, "text_len", text_len);
        if (has_domain && has_text_len && !domain.empty() && text_len > threshold) {
            ++matched_rows;
            sum_text_len += text_len;
        }
    }

    double avg_text_len = matched_rows > 0 ? static_cast<double>(sum_text_len) / matched_rows : 0.0;

    PyObject* result = PyDict_New();
    if (!result) return nullptr;

    PyObject* py_rows = PyLong_FromLongLong(rows);
    PyObject* py_matched = PyLong_FromLongLong(matched_rows);
    PyObject* py_avg = PyFloat_FromDouble(avg_text_len);

    if (!py_rows || !py_matched || !py_avg) {
        Py_XDECREF(py_rows);
        Py_XDECREF(py_matched);
        Py_XDECREF(py_avg);
        Py_DECREF(result);
        return nullptr;
    }

    PyDict_SetItemString(result, "rows", py_rows);
    PyDict_SetItemString(result, "matched_rows", py_matched);
    PyDict_SetItemString(result, "avg_text_len", py_avg);

    Py_DECREF(py_rows);
    Py_DECREF(py_matched);
    Py_DECREF(py_avg);

    return result;
}

static PyMethodDef methods[] = {
    {"scan_jsonl_stats", reinterpret_cast<PyCFunction>(scan_jsonl_stats), METH_VARARGS | METH_KEYWORDS,
     "Scan a JSONL file and compute rows, matched_rows and avg_text_len."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpp_jsonl_fastscan",
    "Fast JSONL scan implemented in C++.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_cpp_jsonl_fastscan(void) {
    return PyModule_Create(&moduledef);
}
