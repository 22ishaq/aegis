// Python bindings (pybind11) for the Aegis core: `import aegis._core`.
//
// The training pipeline calls FeatureEngine.compute_batch on millions of rows,
// so batch entry points take NumPy columns, release the GIL, and loop in C++.
// Per-transaction entry points (used by the API server) take a dict and
// validate it at the boundary before it reaches the hot path.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <vector>

#include "aegis/audit_log.hpp"
#include "aegis/features.hpp"
#include "aegis/mlp.hpp"
#include "aegis/scorer.hpp"
#include "aegis/sha256.hpp"
#include "aegis/types.hpp"

namespace py = pybind11;
using namespace aegis;

namespace {

template <typename T>
using Col = py::array_t<T, py::array::c_style | py::array::forcecast>;

Transaction txn_from_dict(const py::dict& d) {
    Transaction t;
    auto get = [&](const char* k) -> py::object {
        if (!d.contains(k)) throw py::value_error(std::string("transaction is missing field '") + k + "'");
        return d[k];
    };
    t.txn_id = d.contains("txn_id") ? d["txn_id"].cast<uint64_t>() : 0;
    t.account_id = get("account_id").cast<uint64_t>();
    t.merchant_id = get("merchant_id").cast<uint64_t>();
    t.device_id = get("device_id").cast<uint64_t>();
    t.timestamp_ms = get("timestamp_ms").cast<int64_t>();
    t.amount_cents = get("amount_cents").cast<int64_t>();
    t.lat = get("lat").cast<double>();
    t.lon = get("lon").cast<double>();
    t.mcc = get("mcc").cast<uint16_t>();
    t.country = get("country").cast<uint16_t>();
    const int ch = get("channel").cast<int>();
    if (ch < 0 || ch > 3) throw py::value_error("channel must be 0..3");
    t.channel = static_cast<Channel>(ch);
    if (const char* err = validate(t)) throw py::value_error(err);
    return t;
}

// Columnar view over a batch of transactions; checks every column has length n.
struct Batch {
    Col<uint64_t> account, merchant, device;
    Col<int64_t> ts, cents;
    Col<double> lat, lon;
    Col<int64_t> mcc, country, channel;

    py::ssize_t size() const {
        const py::ssize_t n = account.size();
        for (py::ssize_t s : {merchant.size(), device.size(), ts.size(), cents.size(), lat.size(), lon.size(),
                              mcc.size(), country.size(), channel.size()}) {
            if (s != n) throw py::value_error("all columns must have the same length");
        }
        return n;
    }

    Transaction at(py::ssize_t i) const {
        Transaction t;
        t.txn_id = static_cast<uint64_t>(i);
        t.account_id = account.data()[i];
        t.merchant_id = merchant.data()[i];
        t.device_id = device.data()[i];
        t.timestamp_ms = ts.data()[i];
        t.amount_cents = cents.data()[i];
        t.lat = lat.data()[i];
        t.lon = lon.data()[i];
        t.mcc = static_cast<uint16_t>(mcc.data()[i]);
        t.country = static_cast<uint16_t>(country.data()[i]);
        t.channel = static_cast<Channel>(channel.data()[i]);
        return t;
    }

    void validate_all() const {
        const py::ssize_t n = size();
        for (py::ssize_t i = 0; i < n; ++i) {
            if (const char* err = validate(at(i)))
                throw py::value_error("row " + std::to_string(i) + ": " + err);
            if (mcc.data()[i] < 0 || mcc.data()[i] > 65535 || country.data()[i] < 0 || country.data()[i] > 65535)
                throw py::value_error("row " + std::to_string(i) + ": mcc/country out of range");
        }
    }
};

py::list rule_names(uint32_t mask) {
    py::list out;
    for (const RuleInfo& r : kRules) {
        if (mask & r.id) out.append(std::string(r.name));
    }
    return out;
}

py::dict result_to_dict(const ScoreResult& r) {
    py::dict d;
    d["txn_id"] = r.txn_id;
    d["account_id"] = r.account_id;
    d["model_score"] = r.model_score;
    d["risk"] = r.risk;
    d["decision"] = std::string(to_string(r.decision));
    d["rule_mask"] = r.rule_mask;
    d["rules"] = rule_names(r.rule_mask);
    d["hard_rule"] = (r.rule_mask & kHardRules) != 0;
    d["features"] = std::vector<float>(r.features.begin(), r.features.end());
    if (r.has_attribution) {
        d["attribution"] = std::vector<float>(r.attribution.begin(), r.attribution.end());
    } else {
        d["attribution"] = py::none();
    }
    return d;
}

py::bytes digest_bytes(const Digest& d) { return py::bytes(reinterpret_cast<const char*>(d.data()), d.size()); }

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Aegis C++ core: streaming features, MLP inference, rules, crypto, audit chain";

    m.attr("NUM_FEATURES") = kNumFeatures;
    py::list names;
    for (auto n : kFeatureNames) names.append(std::string(n));
    m.attr("FEATURE_NAMES") = names;
    m.attr("HARD_RULES") = kHardRules;
    py::list rules;
    for (const RuleInfo& r : kRules) {
        py::dict d;
        d["id"] = static_cast<uint32_t>(r.id);
        d["name"] = std::string(r.name);
        d["severity"] = r.severity;
        d["description"] = std::string(r.description);
        d["hard"] = (r.id & kHardRules) != 0;
        rules.append(d);
    }
    m.attr("RULES") = rules;

    m.def("rule_names", &rule_names, py::arg("mask"));
    m.def("mcc_risk", &mcc_risk, py::arg("mcc"));

    m.def("sha256", [](py::bytes data) {
        const std::string s = data;
        return digest_bytes(Sha256::hash(s));
    }, py::arg("data"));
    m.def("hmac_sha256", [](py::bytes key, py::bytes msg) {
        const std::string k = key, s = msg;
        return digest_bytes(hmac_sha256(k, s));
    }, py::arg("key"), py::arg("msg"));

    py::register_exception<ModelIntegrityError>(m, "ModelIntegrityError", PyExc_ValueError);

    // ---- Mlp ------------------------------------------------------------------
    py::class_<Mlp>(m, "Mlp")
        .def(py::init([](const std::vector<float>& mean, const std::vector<float>& stddev,
                         const std::vector<std::pair<Col<float>, Col<float>>>& layers, float review, float block) {
                 std::vector<Mlp::Layer> ls;
                 for (const auto& [w, b] : layers) {
                     if (w.ndim() != 2 || b.ndim() != 1) throw py::value_error("layer W must be 2-D and b 1-D");
                     Mlp::Layer L;
                     L.out = static_cast<uint32_t>(w.shape(0));
                     L.in = static_cast<uint32_t>(w.shape(1));
                     L.w.assign(w.data(), w.data() + w.size());
                     L.b.assign(b.data(), b.data() + b.size());
                     ls.push_back(std::move(L));
                 }
                 try {
                     return Mlp(mean, stddev, std::move(ls), review, block);
                 } catch (const std::invalid_argument& e) {
                     throw py::value_error(e.what());
                 }
             }),
             py::arg("mean"), py::arg("std"), py::arg("layers"), py::arg("review_threshold"),
             py::arg("block_threshold"))
        .def_static("from_bytes", [](py::bytes b) {
            const std::string s = b;
            return Mlp::from_bytes(reinterpret_cast<const uint8_t*>(s.data()), s.size());
        })
        .def_static("load", &Mlp::load, py::arg("path"), py::arg("hmac_key") = std::nullopt)
        .def("save", &Mlp::save, py::arg("path"), py::arg("hmac_key") = std::nullopt)
        .def("to_bytes", [](const Mlp& self) {
            const auto v = self.to_bytes();
            return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
        })
        .def("predict_proba", [](const Mlp& self, Col<float> x) {
            if (x.ndim() != 2 || static_cast<std::size_t>(x.shape(1)) != self.n_features())
                throw py::value_error("expected an (N, n_features) float32 array");
            const py::ssize_t n = x.shape(0);
            py::array_t<float> out(n);
            float* o = out.mutable_data();
            const float* p = x.data();
            {
                py::gil_scoped_release release;
                for (py::ssize_t i = 0; i < n; ++i) o[i] = self.predict_proba(p + i * self.n_features());
            }
            return out;
        }, py::arg("x"))
        .def("attribute", [](const Mlp& self, Col<float> x) {
            if (x.ndim() != 1 || static_cast<std::size_t>(x.shape(0)) != self.n_features())
                throw py::value_error("expected a 1-D array of n_features values");
            py::array_t<float> out(static_cast<py::ssize_t>(self.n_features()));
            const float logit = self.attribute(x.data(), out.mutable_data());
            return py::make_tuple(logit, out);
        }, py::arg("x"))
        .def_property_readonly("n_features", &Mlp::n_features)
        .def_property_readonly("review_threshold", &Mlp::review_threshold)
        .def_property_readonly("block_threshold", &Mlp::block_threshold)
        .def_property_readonly("fingerprint", [](const Mlp& self) { return to_hex(self.fingerprint()); })
        .def_property_readonly("mean", &Mlp::mean)
        .def_property_readonly("std", &Mlp::stddev);

    // ---- FeatureEngine ----------------------------------------------------------
    py::class_<FeatureEngine>(m, "FeatureEngine")
        .def(py::init<std::size_t>(), py::arg("expected_accounts") = 1 << 16)
        .def("compute_batch",
             [](FeatureEngine& self, Col<uint64_t> account, Col<uint64_t> merchant, Col<uint64_t> device,
                Col<int64_t> ts, Col<int64_t> cents, Col<double> lat, Col<double> lon, Col<int64_t> mcc,
                Col<int64_t> country, Col<int64_t> channel) {
                 // Rows must be in time order: features are point-in-time by construction.
                 Batch b{account, merchant, device, ts, cents, lat, lon, mcc, country, channel};
                 const py::ssize_t n = b.size();
                 b.validate_all();
                 py::array_t<float> x({n, static_cast<py::ssize_t>(kNumFeatures)});
                 py::array_t<uint32_t> masks(n);
                 float* xp = x.mutable_data();
                 uint32_t* mp = masks.mutable_data();
                 {
                     py::gil_scoped_release release;
                     for (py::ssize_t i = 0; i < n; ++i) {
                         const Transaction t = b.at(i);
                         const FeatureResult f = self.process(t);
                         std::copy(f.x.begin(), f.x.end(), xp + i * static_cast<py::ssize_t>(kNumFeatures));
                         mp[i] = evaluate_rules(t, f.s).mask;
                     }
                 }
                 return py::make_tuple(x, masks);
             },
             py::arg("account_id"), py::arg("merchant_id"), py::arg("device_id"), py::arg("timestamp_ms"),
             py::arg("amount_cents"), py::arg("lat"), py::arg("lon"), py::arg("mcc"), py::arg("country"),
             py::arg("channel"))
        .def_property_readonly("num_accounts", &FeatureEngine::num_accounts)
        .def("reset", &FeatureEngine::reset);

    // ---- Scorer -----------------------------------------------------------------
    py::class_<Scorer>(m, "Scorer")
        .def(py::init<Mlp, std::size_t>(), py::arg("model"), py::arg("expected_accounts") = 1 << 16)
        .def("score", [](Scorer& self, const py::dict& d) { return result_to_dict(self.score(txn_from_dict(d))); },
             py::arg("txn"))
        .def("score_batch",
             [](Scorer& self, Col<uint64_t> account, Col<uint64_t> merchant, Col<uint64_t> device, Col<int64_t> ts,
                Col<int64_t> cents, Col<double> lat, Col<double> lon, Col<int64_t> mcc, Col<int64_t> country,
                Col<int64_t> channel) {
                 Batch b{account, merchant, device, ts, cents, lat, lon, mcc, country, channel};
                 const py::ssize_t n = b.size();
                 b.validate_all();
                 py::array_t<float> model_score(n), risk(n);
                 py::array_t<uint8_t> decision(n);
                 py::array_t<uint32_t> masks(n);
                 float* ms = model_score.mutable_data();
                 float* rk = risk.mutable_data();
                 uint8_t* dc = decision.mutable_data();
                 uint32_t* mk = masks.mutable_data();
                 {
                     py::gil_scoped_release release;
                     for (py::ssize_t i = 0; i < n; ++i) {
                         const ScoreResult r = self.score(b.at(i));
                         ms[i] = r.model_score;
                         rk[i] = r.risk;
                         dc[i] = static_cast<uint8_t>(r.decision);
                         mk[i] = r.rule_mask;
                     }
                 }
                 py::dict out;
                 out["model_score"] = model_score;
                 out["risk"] = risk;
                 out["decision"] = decision;
                 out["rule_mask"] = masks;
                 return out;
             },
             py::arg("account_id"), py::arg("merchant_id"), py::arg("device_id"), py::arg("timestamp_ms"),
             py::arg("amount_cents"), py::arg("lat"), py::arg("lon"), py::arg("mcc"), py::arg("country"),
             py::arg("channel"))
        .def_property_readonly("num_accounts", [](Scorer& self) { return self.engine().num_accounts(); })
        .def_property_readonly("model", &Scorer::model, py::return_value_policy::reference_internal);

    // ---- AuditLog ---------------------------------------------------------------
    auto verify_to_dict = [](const AuditVerifyResult& r) {
        py::dict d;
        d["ok"] = r.ok;
        d["entries"] = r.entries;
        d["first_bad_seq"] = r.first_bad_seq;
        d["reason"] = r.reason;
        d["head"] = r.head_hex;
        return d;
    };
    py::class_<AuditLog>(m, "AuditLog")
        .def(py::init<std::string, std::string, bool>(), py::arg("path"), py::arg("key"), py::arg("fsync") = true)
        .def("append", [](AuditLog& self, const std::string& payload, int64_t ts) {
            py::gil_scoped_release release;  // fsync can take milliseconds
            return self.append(payload, ts);
        }, py::arg("payload"), py::arg("timestamp_ms"))
        .def("verify", [verify_to_dict](const AuditLog& self) { return verify_to_dict(self.verify()); })
        .def_static("verify_file", [verify_to_dict](const std::string& path, const std::string& key) {
            return verify_to_dict(AuditLog::verify_file(path, key));
        }, py::arg("path"), py::arg("key"))
        .def_property_readonly("head", &AuditLog::head_hex)
        .def_property_readonly("size", &AuditLog::size);
}
