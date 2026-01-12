// include/builder/NodeGroup.h
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "builder/Node.h"

namespace sima {

/**
 * @brief Simple wrapper around a linear list of nodes.
 *
 * Builder/Graph are intentionally STL-only; NodeGroup is the shared container
 * between them and the pipeline runtime.
 */
class NodeGroup final {
public:
  using NodePtr = std::shared_ptr<Node>;

  NodeGroup() = default;

  explicit NodeGroup(std::vector<NodePtr>&& nodes) : nodes_(std::move(nodes)) {}

  explicit NodeGroup(const std::vector<NodePtr>& nodes) : nodes_(nodes) {}

  const std::vector<NodePtr>& nodes() const noexcept { return nodes_; }

  std::vector<NodePtr>& nodes_mut() noexcept { return nodes_; }

  bool empty() const noexcept { return nodes_.empty(); }
  std::size_t size() const noexcept { return nodes_.size(); }

private:
  std::vector<NodePtr> nodes_;
};

} // namespace sima
