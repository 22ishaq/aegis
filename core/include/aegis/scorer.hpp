// Rules + model -> one risk score and a decision.
//
// Why both? Rules encode known fraud typologies in business language that a
// regulator can audit line by line, and they work from day one with no training
// data. The model catches subtle combinations no one wrote a rule for. Banks
// run both side by side; Aegis combines them as risk = max(model, rule floor)
// so a rule can raise the score but never hide what the model saw.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "aegis/features.hpp"
#include "aegis/mlp.hpp"
#include "aegis/types.hpp"

namespace aegis {

enum Rule : uint32_t {
    kRuleImpossibleTravel = 1u << 0,
    kRuleVelocityBurst = 1u << 1,
    kRuleCardTesting = 1u << 2,
    kRuleStructuring = 1u << 3,
    kRuleAccountTakeover = 1u << 4,
    kRuleHighRiskLarge = 1u << 5,
};

// "Hard" rules describe patterns specific enough that an alert carrying one
// must be looked at by a human: no automated system (including the LLM
// investigator) is allowed to dismiss it on its own.
inline constexpr uint32_t kHardRules =
    kRuleImpossibleTravel | kRuleCardTesting | kRuleStructuring | kRuleAccountTakeover;

struct RuleInfo {
    Rule id;
    std::string_view name;
    float severity;  // minimum risk score when the rule fires
    std::string_view description;
};

inline constexpr std::array<RuleInfo, 6> kRules = {{
    {kRuleImpossibleTravel, "IMPOSSIBLE_TRAVEL", 0.95f,
     "Card used >500 km from its previous location at an implied speed above 900 km/h"},
    {kRuleVelocityBurst, "VELOCITY_BURST", 0.70f, "8 or more transactions on the account in the past hour"},
    {kRuleCardTesting, "CARD_TESTING", 0.90f,
     "3 or more sub-$5 transactions in 10 minutes: probing whether a stolen card works"},
    {kRuleStructuring, "STRUCTURING", 0.85f,
     "Repeated amounts just under the $10,000 reporting threshold within 24 hours"},
    {kRuleAccountTakeover, "ACCOUNT_TAKEOVER", 0.90f,
     "Unrecognised device in a new country spending far above the account's norm"},
    {kRuleHighRiskLarge, "HIGH_RISK_MERCHANT_LARGE", 0.60f,
     "$2,000+ at a high-risk merchant category (gambling, crypto, wires)"},
}};

struct RuleResult {
    uint32_t mask = 0;
    float severity = 0.0f;  // max severity among fired rules
};

RuleResult evaluate_rules(const Transaction& t, const Signals& s) noexcept;

struct ScoreResult {
    uint64_t txn_id = 0;
    uint64_t account_id = 0;
    float model_score = 0.0f;  // network output probability
    float risk = 0.0f;         // max(model_score, rule severity)
    Decision decision = Decision::Allow;
    uint32_t rule_mask = 0;
    bool has_attribution = false;
    FeatureVector features{};
    std::array<float, kNumFeatures> attribution{};  // filled only for REVIEW/BLOCK
};

class Scorer {
public:
    // The model must take exactly kNumFeatures inputs.
    explicit Scorer(Mlp model, std::size_t expected_accounts = 1 << 16);

    // Features -> model -> rules -> decision. Explanations (a backward pass)
    // are computed only for flagged transactions: ~99% of traffic is allowed
    // and should not pay for an explanation no one will read.
    ScoreResult score(const Transaction& t);

    const Mlp& model() const noexcept { return model_; }
    FeatureEngine& engine() noexcept { return engine_; }

private:
    Mlp model_;
    FeatureEngine engine_;
};

}  // namespace aegis
