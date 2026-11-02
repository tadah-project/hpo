#pragma once
/*
 * The refusals of the lines of a target file name the file as --hpotarget
 * gives it: "line 8 of /tmp/tadah_svs_4711_3_99.hpo: <what>\nLine: <text>".
 * A test that writes its target file under a temporary name compares such
 * a text with the file written as <target>: as_target(text, path) is
 * `text` with every occurrence of `path` replaced by "<target>", so that
 * "line 8 of <target>: ..." pins the line, the file and the message.
 */

#include "catch2/catch.hpp"

#include <string>
#include <utility>

namespace hpo_test {

inline std::string as_target(std::string text, const std::string& path)
{
    if (path.empty()) return text;
    const std::string name = "<target>";
    for (std::size_t p = text.find(path); p != std::string::npos;
         p = text.find(path, p + name.size()))
        text.replace(p, path.size(), name);
    return text;
}

/* A matcher of an exception's message for CHECK_THROWS_WITH: the message
 * with `path` written as <target> (as_target) equals `expected`, or
 * matches `inner`. The inner matcher is held by reference: it lives as
 * long as the assertion that builds both. */
class AsTarget : public Catch::MatcherBase<std::string> {
public:
    AsTarget(std::string path, std::string expected)
        : path_(std::move(path)), expected_(std::move(expected)) {}
    AsTarget(std::string path, const Catch::MatcherBase<std::string>& inner)
        : path_(std::move(path)), inner_(&inner) {}
    bool match(const std::string& message) const override
    {
        const std::string named = as_target(message, path_);
        return inner_ ? inner_->match(named) : named == expected_;
    }
    std::string describe() const override
    {
        return "with " + path_ + " written as <target>, " +
               (inner_ ? inner_->toString() : "equals \"" + expected_ + "\"");
    }

private:
    std::string path_;
    std::string expected_;
    const Catch::MatcherBase<std::string>* inner_ = nullptr;
};

}  // namespace hpo_test
