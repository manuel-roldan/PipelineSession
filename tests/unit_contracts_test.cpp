#include "contracts/ContractRegistry.h"
#include "contracts/Validators.h"

#include "test_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class FakeNode final : public sima::Node {
public:
  FakeNode(std::string kind, std::string label)
      : kind_(std::move(kind)), label_(std::move(label)) {}

  std::string kind() const override { return kind_; }
  std::string user_label() const override { return label_; }
  std::string gst_fragment(int) const override { return "identity"; }
  std::vector<std::string> element_names(int) const override { return {}; }

private:
  std::string kind_;
  std::string label_;
};

} // namespace

int main() {
  try {
    using sima::ContractRegistry;
    using sima::NodeGroup;
    using sima::ValidationContext;

    ContractRegistry reg;
    reg.add(sima::validators::NonEmptyPipeline());
    reg.add(sima::validators::NoNullNodes());
    reg.add(sima::validators::SinkLastForRun());
    reg.add(sima::validators::TapPointExists());
    reg.add(sima::validators::UniqueDebugPointLabels());

    ValidationContext ctx;
    ctx.mode = ValidationContext::Mode::Validate;
    ctx.strict = true;

    NodeGroup empty;
    auto rep = reg.validate(empty, ctx);
    require(rep.has_errors(), "Empty pipeline should fail");

    auto dbg = std::make_shared<FakeNode>("DebugPoint", "tap");
    auto sink = std::make_shared<FakeNode>("OutputAppSink", "");
    auto other = std::make_shared<FakeNode>("Other", "");

    NodeGroup run_ok({other, sink});
    ctx.mode = ValidationContext::Mode::Run;
    rep = reg.validate(run_ok, ctx);
    require(!rep.has_errors(), "Run mode should accept sink last");

    ctx.mode = ValidationContext::Mode::Tap;
    ctx.tap_name = "tap";
    NodeGroup tap_ok({dbg, other});
    rep = reg.validate(tap_ok, ctx);
    require(!rep.has_errors(), "Tap should find DebugPoint");

    auto dbg2 = std::make_shared<FakeNode>("DebugPoint", "tap");
    NodeGroup dup({dbg, dbg2});
    ctx.mode = ValidationContext::Mode::Validate;
    rep = reg.validate(dup, ctx);
    require(rep.warning_count() > 0 || rep.error_count() > 0,
            "Duplicate DebugPoint labels should warn");

    std::cout << "[OK] unit_contracts_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
