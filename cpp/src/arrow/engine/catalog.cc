// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/engine/catalog.h"

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/table.h"

#include "arrow/dataset/dataset.h"

namespace arrow {
namespace engine {

//
// Catalog
//

using Entry = Catalog::Entry;

Catalog::Catalog(std::unordered_map<std::string, Entry> datasets)
    : datasets_(std::move(datasets)) {}

Result<Entry> Catalog::Get(const std::string& key) const {
  auto value = datasets_.find(key);
  if (value != datasets_.end()) return value->second;
  return Status::KeyError("Table '", key, "' not found in catalog.");
}

Result<std::shared_ptr<Schema>> Catalog::GetSchema(const std::string& key) const {
  auto as_schema = [](const Entry& entry) -> Result<std::shared_ptr<Schema>> {
    return entry.schema();
  };
  return Get(key).Map(as_schema);
}

Result<std::shared_ptr<Catalog>> Catalog::Make(const std::vector<Entry>& datasets) {
  CatalogBuilder builder;

  for (const auto& key_val : datasets) {
    RETURN_NOT_OK(builder.Add(key_val));
  }

  return builder.Finish();
}

//
// Catalog::Entry
//

Entry::Entry(std::shared_ptr<dataset::Dataset> dataset, std::string name)
    : dataset_(std::move(dataset)), name_(std::move(name)) {}

Entry::Entry(std::shared_ptr<Table> table, std::string name)
    : dataset_(std::make_shared<dataset::InMemoryDataset>(std::move(table))),
      name_(std::move(name)) {}

const std::shared_ptr<dataset::Dataset>& Entry::dataset() const { return dataset_; }

bool Entry::operator==(const Entry& other) const {
  // Entries are unique by name in a catalog, but we can still protect with
  // pointer equality.
  return name_ == other.name_ && dataset_ == other.dataset_;
}

const std::shared_ptr<Schema>& Entry::schema() const { return dataset()->schema(); }

//
// CatalogBuilder
//

Status CatalogBuilder::Add(Entry entry) {
  const auto& name = entry.name();
  if (name.empty()) {
    return Status::Invalid("Key in catalog can't be empty");
  }

  if (entry.dataset() == nullptr) {
    return Status::Invalid("Dataset entry can't be null.");
  }

  auto inserted = datasets_.emplace(name, std::move(entry));
  if (!inserted.second) {
    return Status::KeyError("Dataset '", name, "' already in catalog.");
  }

  return Status::OK();
}

Status CatalogBuilder::Add(std::string name, std::shared_ptr<dataset::Dataset> dataset) {
  return Add(Entry(std::move(dataset), std::move(name)));
}

Status CatalogBuilder::Add(std::string name, std::shared_ptr<Table> table) {
  return Add(Entry(std::move(table), std::move(name)));
}

Result<std::shared_ptr<Catalog>> CatalogBuilder::Finish() {
  return std::shared_ptr<Catalog>(new Catalog(std::move(datasets_)));
}

}  // namespace engine
}  // namespace arrow
