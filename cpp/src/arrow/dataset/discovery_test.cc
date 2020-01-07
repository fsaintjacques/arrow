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

#include "arrow/dataset/discovery.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "arrow/dataset/partition.h"
#include "arrow/dataset/test_util.h"
#include "arrow/filesystem/test_util.h"

namespace arrow {
namespace dataset {

class SourceManifestTest : public TestFileSystemSource {
 public:
  void AssertInspect(const std::vector<std::shared_ptr<Field>>& expected_fields) {
    ASSERT_OK_AND_ASSIGN(auto actual, manifest_->Inspect());
    EXPECT_EQ(*actual, Schema(expected_fields));
  }

  void AssertInspectSchemas(std::vector<std::shared_ptr<Schema>> expected) {
    ASSERT_OK_AND_ASSIGN(auto actual, manifest_->InspectSchemas());

    EXPECT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); i++) {
      EXPECT_EQ(*actual[i], *expected[i]);
    }
  }

 protected:
  std::shared_ptr<SourceManifest> manifest_;
};

class MockSourceManifest : public SourceManifest {
 public:
  explicit MockSourceManifest(std::vector<std::shared_ptr<Schema>> schemas)
      : schemas_(std::move(schemas)) {}

  Result<std::vector<std::shared_ptr<Schema>>> InspectSchemas() override {
    return schemas_;
  }

  Result<std::shared_ptr<Source>> Finish(const std::shared_ptr<Schema>&) override {
    return std::make_shared<SimpleSource>(std::vector<std::shared_ptr<Fragment>>{});
  }

 protected:
  std::vector<std::shared_ptr<Schema>> schemas_;
};

class MockPartitionScheme : public PartitionScheme {
 public:
  explicit MockPartitionScheme(std::shared_ptr<Schema> schema)
      : PartitionScheme(std::move(schema)) {}

  Result<std::shared_ptr<Expression>> Parse(const std::string& segment,
                                            int i) const override {
    return nullptr;
  }

  std::string type_name() const override { return "mock_partition_scheme"; }
};

class MockSourceManifestTest : public SourceManifestTest {
 public:
  void MakeManifest(std::vector<std::shared_ptr<Schema>> schemas) {
    manifest_ = std::make_shared<MockSourceManifest>(schemas);
  }

 protected:
  std::shared_ptr<Field> i32 = field("i32", int32());
  std::shared_ptr<Field> i64 = field("i64", int64());
  std::shared_ptr<Field> f32 = field("f32", float64());
  std::shared_ptr<Field> f64 = field("f64", float64());
  // Non-nullable
  std::shared_ptr<Field> i32_req = field("i32", int32(), false);
  // bad type with name `i32`
  std::shared_ptr<Field> i32_fake = field("i32", boolean());
};

TEST_F(MockSourceManifestTest, UnifySchemas) {
  MakeManifest({});
  AssertInspect({});

  MakeManifest({schema({i32}), schema({i32})});
  AssertInspect({i32});

  MakeManifest({schema({i32}), schema({i64})});
  AssertInspect({i32, i64});

  MakeManifest({schema({i32}), schema({i64})});
  AssertInspect({i32, i64});

  MakeManifest({schema({i32}), schema({i32_req})});
  AssertInspect({i32});

  MakeManifest({schema({i32, f64}), schema({i32_req, i64})});
  AssertInspect({i32, f64, i64});

  MakeManifest({schema({i32, f64}), schema({f64, i32_fake})});
  // Unification fails when fields with the same name have clashing types.
  ASSERT_RAISES(Invalid, manifest_->Inspect());
  // Return the individual schema for closer inspection should not fail.
  AssertInspectSchemas({schema({i32, f64}), schema({f64, i32_fake})});
}

class FileSystemSourceManifestTest : public SourceManifestTest {
 public:
  void MakeManifest(const std::vector<fs::FileStats>& files) {
    MakeFileSystem(files);
    ASSERT_OK_AND_ASSIGN(manifest_, FileSystemSourceManifest::Make(
                                        fs_, selector_, format_, manifest_options_));
  }

  void AssertFinishWithPaths(std::vector<std::string> paths,
                             std::shared_ptr<Schema> schema = nullptr) {
    if (schema == nullptr) {
      ASSERT_OK_AND_ASSIGN(schema, manifest_->Inspect());
    }
    options_ = ScanOptions::Make(schema);
    ASSERT_OK_AND_ASSIGN(source_, manifest_->Finish(schema));
    AssertFragmentsAreFromPath(source_->GetFragments(options_), paths);
  }

 protected:
  fs::FileSelector selector_;
  FileSystemManifestOptions manifest_options_;
  std::shared_ptr<FileFormat> format_ = std::make_shared<DummyFileFormat>(schema({}));
};

TEST_F(FileSystemSourceManifestTest, Basic) {
  MakeManifest({fs::File("a"), fs::File("b")});
  AssertFinishWithPaths({"a", "b"});
  MakeManifest({fs::Dir("a"), fs::Dir("a/b"), fs::File("a/b/c")});
}

TEST_F(FileSystemSourceManifestTest, Selector) {
  selector_.base_dir = "A";
  selector_.recursive = true;

  MakeManifest({fs::File("0"), fs::File("A/a"), fs::File("A/A/a")});
  // "0" doesn't match selector, so it has been dropped:
  AssertFinishWithPaths({"A/a", "A/A/a"});

  manifest_options_.partition_base_dir = "A/A";
  MakeManifest({fs::File("0"), fs::File("A/a"), fs::File("A/A/a")});
  // partition_base_dir should not affect filtered files, ony the applied
  // partition scheme.
  AssertInspect({});
  AssertFinishWithPaths({"A/a", "A/A/a"});
}

TEST_F(FileSystemSourceManifestTest, ExplicitPartition) {
  selector_.base_dir = "a=ignored/base";
  manifest_options_.partition_scheme =
      std::make_shared<HivePartitionScheme>(schema({field("a", float64())}));

  MakeManifest(
      {fs::File(selector_.base_dir + "/a=1"), fs::File(selector_.base_dir + "/a=2")});

  AssertInspect({field("a", float64())});
  AssertFinishWithPaths({selector_.base_dir + "/a=1", selector_.base_dir + "/a=2"});
}

TEST_F(FileSystemSourceManifestTest, DiscoveredPartition) {
  selector_.base_dir = "a=ignored/base";
  manifest_options_.partition_scheme = HivePartitionScheme::MakeManifest();
  MakeManifest(
      {fs::File(selector_.base_dir + "/a=1"), fs::File(selector_.base_dir + "/a=2")});

  AssertInspect({field("a", int32())});
  AssertFinishWithPaths({selector_.base_dir + "/a=1", selector_.base_dir + "/a=2"});
}

TEST_F(FileSystemSourceManifestTest, MissingDirectories) {
  MakeFileSystem({fs::File("base_dir/a=3/b=3/dat"), fs::File("unpartitioned/ignored=3")});

  manifest_options_.partition_base_dir = "base_dir";
  manifest_options_.partition_scheme = std::make_shared<HivePartitionScheme>(
      schema({field("a", int32()), field("b", int32())}));

  ASSERT_OK_AND_ASSIGN(
      manifest_, FileSystemSourceManifest::Make(
                     fs_, {"base_dir/a=3/b=3/dat", "unpartitioned/ignored=3"}, format_,
                     manifest_options_));

  AssertInspect({field("a", int32()), field("b", int32())});
  AssertFinishWithPaths({"base_dir/a=3/b=3/dat", "unpartitioned/ignored=3"});
}

TEST_F(FileSystemSourceManifestTest, OptionsIgnoredDefaultPrefixes) {
  MakeManifest({
      fs::File("."),
      fs::File("_"),
      fs::File("_$folder$"),
      fs::File("_SUCCESS"),
      fs::File("not_ignored_by_default"),
  });

  AssertFinishWithPaths({"not_ignored_by_default"});
}

TEST_F(FileSystemSourceManifestTest, OptionsIgnoredCustomPrefixes) {
  manifest_options_.ignore_prefixes = {"not_ignored"};
  MakeManifest({
      fs::File("."),
      fs::File("_"),
      fs::File("_$folder$"),
      fs::File("_SUCCESS"),
      fs::File("not_ignored_by_default"),
  });

  AssertFinishWithPaths({".", "_", "_$folder$", "_SUCCESS"});
}

TEST_F(FileSystemSourceManifestTest, Inspect) {
  auto s = schema({field("f64", float64())});
  format_ = std::make_shared<DummyFileFormat>(s);

  // No files
  MakeManifest({});
  AssertInspect({});

  MakeManifest({fs::File("test")});
  AssertInspect(s->fields());
}

}  // namespace dataset
}  // namespace arrow
