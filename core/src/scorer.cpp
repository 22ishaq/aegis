#include "aegis/scorer.hpp"

#include <algorithm>
#include <stdexcept>

namespace aegis {

namespace {

float severity_of(uint32_t mask) noexcept {
    float s = 0.0f;
    for (const RuleInfo& r : kRules) {
        if (mask & r.id) s = std::max(s, r.severity);
    }
    return s;
}

}  // namespace

RuleResult evaluate_rules(const Transaction& t, const Signals& s) noexcept {
    uint32_t m = 0;
    if (!s.first_txn && s.km_from_last > 500.0 && s.speed_kmh > 900.0) m |= kRuleImpossibleTravel;
    if (s.txn_1h >= 8.0) m |= kRuleVelocityBurst;
    if (s.small_10m >= 3.0) m |= kRuleCardTesting;
    const bool near_threshold = t.amount_cents >= 900000 && t.amount_cents < 1000000;
    if (near_threshold && s.near_threshold_24h >= 0.6) m |= kRuleStructuring;
    if (!s.first_txn && s.new_device && s.new_country && s.amount_z > 2.0) m |= kRuleAccountTakeover;
    if (s.mcc_risk >= 0.8 && s.amount_usd >= 2000.0) m |= kRuleHighRiskLarge;
    return {m, severity_of(m)};
}

Scorer::Scorer(Mlp model, std::size_t expected_accounts) : model_(std::move(model)), engine_(expected_accounts) {
    if (model_.n_features() != kNumFeatures) {
        throw std::invalid_argument("model expects " + std::to_string(model_.n_features()) +
                                    " features but the engine produces " + std::to_string(kNumFeatures));
    }
}

ScoreResult Scorer::score(const Transaction& t) {
    const FeatureResult f = engine_.process(t);
    const RuleResult rules = evaluate_rules(t, f.s);

    ScoreResult r;
    r.txn_id = t.txn_id;
    r.account_id = t.account_id;
    r.features = f.x;
    r.rule_mask = rules.mask;
    r.model_score = model_.predict_proba(f.x.data());
    r.risk = std::max(r.model_score, rules.severity);

    if (r.risk >= model_.block_threshold()) {
        r.decision = Decision::Block;
    } else if (r.risk >= model_.review_threshold() || (rules.mask & kHardRules)) {
        r.decision = Decision::Review;
    }

    if (r.decision != Decision::Allow) {
        model_.attribute(f.x.data(), r.attribution.data());
        r.has_attribution = true;
    }
    return r;
}

}  // namespace aegis
