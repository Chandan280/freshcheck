// grader.h - turns p_rotten into grade, action, shelf-life estimate and messages.
#pragma once
#include <string>

#include "json.hpp"

struct Grade {
    std::string grade;         // "Fresh" | "Medium" | "Rotten"
    float confidence = 0;      // how sure we are of fresh-vs-rotten: max(p, 1-p)
    std::string action;        // English action text
    std::string action_color;  // "green" | "amber" | "red"
    int use_within_days = 0;   // PROTOTYPE estimate
    std::string message_en, message_hi;
};

class Grader {
public:
    explicit Grader(const nlohmann::json& cfg) : cfg_(cfg) {}
    Grade grade(const std::string& fruit, float p_rotten) const;

    // Estimated kg and rupees for one logged fruit (0 if its grade is not "saved").
    double saved_kg(const std::string& fruit, const std::string& grade) const;
    double saved_inr(const std::string& fruit, const std::string& grade) const;

private:
    nlohmann::json cfg_;  // full config
};
