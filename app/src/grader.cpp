// grader.cpp - simple, transparent rules read from config.json.
#include "grader.h"

#include <algorithm>
#include <cmath>

Grade Grader::grade(const std::string& fruit, float p_rotten) const {
    const auto& g = cfg_.at("grading");
    Grade out;
    if (p_rotten < g.value("fresh_below", 0.35f)) out.grade = "Fresh";
    else if (p_rotten > g.value("rotten_above", 0.65f)) out.grade = "Rotten";
    else out.grade = "Medium";
    out.confidence = std::max(p_rotten, 1.f - p_rotten);

    // Shelf life (prototype): base days scaled slightly by how fresh the model thinks it is.
    const auto& sl = cfg_.at("shelf_life");
    float base = sl.at("base_days").value(out.grade, 0.f);
    float k = sl.value("confidence_k", 0.3f);
    out.use_within_days = (int)std::lround(base * (1.f - k + k * (1.f - p_rotten)));
    if (out.grade == "Medium") out.use_within_days = std::max(out.use_within_days, 1);

    const auto& act = cfg_.at("actions").at(out.grade);
    out.action = act.value("en", "");
    out.action_color = act.value("color", "");

    // Bilingual one-line messages.
    std::string fruit_hi = cfg_.value("fruit_names_hi", nlohmann::json::object()).value(fruit, fruit);
    std::string days = std::to_string(out.use_within_days);
    // Advice part of the action text: "Fresh - sell at full price" -> "sell at full price".
    auto advice = [](const std::string& a) { auto p = a.find(" - "); return p == std::string::npos ? a : a.substr(p + 3); };
    if (out.grade == "Rotten") {
        out.message_en = "Rotten " + fruit + ": don't sell or donate. Compost it.";
        out.message_hi = fruit_hi + " सड़ा हुआ है: न बेचें, न दान करें। खाद बनाएँ।";
    } else {
        out.message_en = out.grade + " " + fruit + ": " + advice(out.action) + ". Use within ~" + days + " day(s) (prototype estimate).";
        out.message_hi = fruit_hi + ": " + advice(act.value("hi", "")) + "। लगभग " + days + " दिन में इस्तेमाल करें (अनुमान)।";
    }
    return out;
}

double Grader::saved_kg(const std::string& fruit, const std::string& grade) const {
    const auto& s = cfg_.at("savings");
    const auto& grades = s.at("saved_grades");
    if (std::find(grades.begin(), grades.end(), grade) == grades.end()) return 0;
    return s.at("avg_weight_kg").value(fruit, 0.15);
}

double Grader::saved_inr(const std::string& fruit, const std::string& grade) const {
    return saved_kg(fruit, grade) * cfg_.at("savings").at("price_inr_per_kg").value(fruit, 80.0);
}
