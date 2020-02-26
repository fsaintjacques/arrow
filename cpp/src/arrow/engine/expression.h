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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "arrow/engine/catalog.h"
#include "arrow/type_fwd.h"
#include "arrow/util/variant.h"

namespace arrow {
namespace engine {

// Expression
class ARROW_EXPORT ExprType {
 public:
  enum Shape {
    SCALAR,
    ARRAY,
    TABLE,
  };

  static ExprType Scalar(std::shared_ptr<DataType> type);
  static ExprType Array(std::shared_ptr<DataType> type);
  static ExprType Table(std::shared_ptr<Schema> schema);

  /// \brief Shape of the expression.
  Shape shape() const { return shape_; }
  /// \brief Schema of the expression if of table shape.
  std::shared_ptr<Schema> schema() const;
  std::shared_ptr<DataType> data_type() const;

  bool Equals(const ExprType& type) const;
  bool operator==(const ExprType& rhs) const;

  std::string ToString() const;

 private:
  ExprType(std::shared_ptr<Schema> schema, Shape shape);
  ExprType(std::shared_ptr<DataType> type, Shape shape);

  util::variant<std::shared_ptr<DataType>, std::shared_ptr<Schema>> type_;
  Shape shape_;
};

/// Represents an expression tree
class ARROW_EXPORT Expr {
 public:
  enum Kind {
    //
    SCALAR,
    FIELD_REF,
    //
    EQ_OP,
    //
    SCAN_REL,
    FILTER_REL,
  };

  Kind kind() const { return kind_; }
  virtual ExprType type() const = 0;

  /// Returns true iff the expressions are identical; does not check for equivalence.
  /// For example, (A and B) is not equal to (B and A) nor is (A and not A) equal to
  /// (false).
  bool Equals(const Expr& other) const;
  bool Equals(const std::shared_ptr<Expr>& other) const;

  /// Return a string representing the expression
  std::string ToString() const;

  virtual ~Expr() = default;

 protected:
  explicit Expr(Kind kind) : kind_(kind) {}

 private:
  Kind kind_;
};

//
// Value Expressions
//

// An unnamed scalar literal expression.
class ScalarExpr : public Expr {
 public:
  static Result<std::shared_ptr<ScalarExpr>> Make(std::shared_ptr<Scalar> scalar);

  const std::shared_ptr<Scalar>& scalar() const { return scalar_; }

  ExprType type() const override;

 private:
  explicit ScalarExpr(std::shared_ptr<Scalar> scalar);
  std::shared_ptr<Scalar> scalar_;
};

// References a column in a table/dataset
class FieldRefExpr : public Expr {
 public:
  static Result<std::shared_ptr<FieldRefExpr>> Make(std::shared_ptr<Field> field);

  const std::shared_ptr<Field>& field() const { return field_; }

  ExprType type() const override;

 private:
  explicit FieldRefExpr(std::shared_ptr<Field> field);

  std::shared_ptr<Field> field_;
};

//
// Operators expression
//

using ExprVector = std::vector<std::shared_ptr<Expr>>;

class OpExpr {
 public:
  const ExprVector& inputs() const { return inputs_; }

 protected:
  explicit OpExpr(ExprVector inputs) : inputs_(std::move(inputs)) {}
  ExprVector inputs_;
};

template <Expr::Kind KIND>
class BinaryOpExpr : public Expr, private OpExpr {
 public:
  const std::shared_ptr<Expr>& left() const { return inputs_[0]; }
  const std::shared_ptr<Expr>& right() const { return inputs_[1]; }

 protected:
  BinaryOpExpr(std::shared_ptr<Expr> left, std::shared_ptr<Expr> right)
      : Expr(KIND), OpExpr({std::move(left), std::move(right)}) {}
};

class EqOpExpr : public BinaryOpExpr<Expr::EQ_OP> {};

//
// Relational Expressions
//

class ScanRelExpr : public Expr {
 public:
  static Result<std::shared_ptr<Expr>> Make(Catalog::Entry input);

  ExprType type() const override;

 private:
  explicit ScanRelExpr(Catalog::Entry input);

  Catalog::Entry input_;
};

class FilterRelExpr : public Expr {
 public:
  static Result<std::shared_ptr<Expr>> Make(std::shared_ptr<Expr> input,
                                            std::shared_ptr<Expr> predicate);

  ExprType type() const override;

 private:
  FilterRelExpr(std::shared_ptr<Expr> input, std::shared_ptr<Expr> predicate);

  std::shared_ptr<Expr> input_;
  std::shared_ptr<Expr> predicate_;
};

}  // namespace engine
}  // namespace arrow
