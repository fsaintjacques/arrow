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

// Object model for scalar (non-Array) values. Not intended for use with large
// amounts of data
//
// NOTE: This API is experimental as of the 0.13 version and subject to change
// without deprecation warnings

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "arrow/type_traits.h"
#include "arrow/util/decimal.h"
#include "arrow/util/logging.h"
#include "arrow/util/string_view.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;

/// \brief Base class for scalar values, representing a single value occupying
/// an array "slot"
struct ARROW_EXPORT Scalar {
  virtual ~Scalar() = default;

  explicit Scalar(const std::shared_ptr<DataType>& type) : Scalar(type, false) {}

  /// \brief The type of the scalar value
  std::shared_ptr<DataType> type;

  /// \brief Whether the value is valid (not null) or not
  bool is_valid = false;

  bool Equals(const Scalar& other) const;
  bool Equals(const std::shared_ptr<Scalar>& other) const {
    if (other) return Equals(*other);
    return false;
  }

  std::string ToString() const;

  static Result<std::shared_ptr<Scalar>> Parse(const std::shared_ptr<DataType>& type,
                                               util::string_view repr);

  // TODO(bkietz) add compute::CastOptions
  Result<std::shared_ptr<Scalar>> CastTo(std::shared_ptr<DataType> to) const;

 protected:
  Scalar(const std::shared_ptr<DataType>& type, bool is_valid)
      : type(type), is_valid(is_valid) {}
};

/// \brief A scalar value for NullType. Never valid
struct ARROW_EXPORT NullScalar : public Scalar {
 public:
  using TypeClass = NullType;

  NullScalar() : Scalar{null(), false} {}
};

namespace internal {

template <typename T, typename CType = typename T::c_type>
struct ARROW_EXPORT PrimitiveScalar : public Scalar {
  using Scalar::Scalar;
  using TypeClass = T;
  using ValueType = CType;

  // Non-null constructor.
  PrimitiveScalar(ValueType value, const std::shared_ptr<DataType>& type)
      : Scalar(type, true), value(value) {
    ARROW_CHECK_EQ(type->id(), T::type_id);
  }

  // Non-null constructor without type's pointer if the type is parameter-free.
  template <typename T1 = T>
  explicit PrimitiveScalar(
      ValueType value,
      typename std::enable_if<TypeTraits<T1>::is_parameter_free>::type* = NULLPTR)
      : PrimitiveScalar(value, TypeTraits<T>::type_singleton()) {}

  // Null constructor without type's pointer if the type is parameter-free.
  template <typename T1 = T>
  PrimitiveScalar(
      typename std::enable_if<TypeTraits<T1>::is_parameter_free>::type* = NULLPTR)
      : Scalar(TypeTraits<T>::type_singleton()) {}

  ValueType value;
};

}  // namespace internal

struct ARROW_EXPORT BooleanScalar : public internal::PrimitiveScalar<BooleanType, bool> {
  using internal::PrimitiveScalar<BooleanType, bool>::PrimitiveScalar;
};

template <typename T>
struct ARROW_EXPORT NumericScalar : public internal::PrimitiveScalar<T> {
  using internal::PrimitiveScalar<T>::PrimitiveScalar;
};

struct ARROW_EXPORT BaseBinaryScalar : public Scalar {
  using Scalar::Scalar;
  using ValueType = std::shared_ptr<Buffer>;

  std::shared_ptr<Buffer> value;

 protected:
  BaseBinaryScalar(const std::shared_ptr<Buffer>& value,
                   const std::shared_ptr<DataType>& type)
      : Scalar{type, true}, value(value) {}
};

struct ARROW_EXPORT BinaryScalar : public BaseBinaryScalar {
  using BaseBinaryScalar::BaseBinaryScalar;
  using TypeClass = BinaryScalar;

  BinaryScalar(const std::shared_ptr<Buffer>& value,
               const std::shared_ptr<DataType>& type)
      : BaseBinaryScalar(value, type) {}

  explicit BinaryScalar(const std::shared_ptr<Buffer>& value)
      : BinaryScalar(value, binary()) {}

  BinaryScalar() : BinaryScalar(binary()) {}
};

struct ARROW_EXPORT StringScalar : public BinaryScalar {
  using BinaryScalar::BinaryScalar;
  using TypeClass = StringType;

  explicit StringScalar(const std::shared_ptr<Buffer>& value)
      : StringScalar(value, utf8()) {}

  explicit StringScalar(std::string s);

  StringScalar() : StringScalar(utf8()) {}
};

struct ARROW_EXPORT LargeBinaryScalar : public BaseBinaryScalar {
  using BaseBinaryScalar::BaseBinaryScalar;
  using TypeClass = LargeBinaryScalar;

  LargeBinaryScalar(const std::shared_ptr<Buffer>& value,
                    const std::shared_ptr<DataType>& type)
      : BaseBinaryScalar(value, type) {}

  explicit LargeBinaryScalar(const std::shared_ptr<Buffer>& value)
      : LargeBinaryScalar(value, large_binary()) {}

  LargeBinaryScalar() : LargeBinaryScalar(large_binary()) {}
};

struct ARROW_EXPORT LargeStringScalar : public LargeBinaryScalar {
  using LargeBinaryScalar::LargeBinaryScalar;
  using TypeClass = LargeStringScalar;

  explicit LargeStringScalar(const std::shared_ptr<Buffer>& value)
      : LargeStringScalar(value, large_utf8()) {}

  LargeStringScalar() : LargeStringScalar(large_utf8()) {}
};

struct ARROW_EXPORT FixedSizeBinaryScalar : public BinaryScalar {
  using TypeClass = FixedSizeBinaryType;

  FixedSizeBinaryScalar(const std::shared_ptr<Buffer>& value,
                        const std::shared_ptr<DataType>& type);

  explicit FixedSizeBinaryScalar(const std::shared_ptr<DataType>& type)
      : BinaryScalar(type) {}
};

template <typename T>
struct ARROW_EXPORT TemporalScalar : public internal::PrimitiveScalar<T> {
  using TypeClass = T;
  using internal::PrimitiveScalar<T>::PrimitiveScalar;
};

template <typename T>
struct ARROW_EXPORT DateScalar : public TemporalScalar<T> {
  using TemporalScalar<T>::TemporalScalar;
};

template <typename T>
struct ARROW_EXPORT TimeScalar : public TemporalScalar<T> {
  using TemporalScalar<T>::TemporalScalar;
};

template <typename T>
struct ARROW_EXPORT IntervalScalar : public internal::PrimitiveScalar<T> {
  using internal::PrimitiveScalar<T>::PrimitiveScalar;
};

struct ARROW_EXPORT Decimal128Scalar : public Scalar {
  using Scalar::Scalar;
  using TypeClass = Decimal128Type;
  using ValueType = Decimal128;

  Decimal128Scalar(const Decimal128& value, const std::shared_ptr<DataType>& type)
      : Scalar(type, true), value(std::move(value)) {}

  Decimal128 value;
};

struct ARROW_EXPORT BaseListScalar : public Scalar {
  using Scalar::Scalar;
  using ValueType = std::shared_ptr<Array>;

  BaseListScalar(const std::shared_ptr<Array>& value,
                 const std::shared_ptr<DataType>& type);

  explicit BaseListScalar(const std::shared_ptr<Array>& value);

  std::shared_ptr<Array> value;
};

struct ARROW_EXPORT ListScalar : public BaseListScalar {
  using TypeClass = ListType;
  using BaseListScalar::BaseListScalar;
};

struct ARROW_EXPORT LargeListScalar : public BaseListScalar {
  using TypeClass = LargeListType;
  using BaseListScalar::BaseListScalar;
};

struct ARROW_EXPORT MapScalar : public BaseListScalar {
  using TypeClass = MapType;
  using BaseListScalar::BaseListScalar;
};

struct ARROW_EXPORT FixedSizeListScalar : public BaseListScalar {
  using TypeClass = FixedSizeListType;
  using BaseListScalar::BaseListScalar;

  FixedSizeListScalar(const std::shared_ptr<Array>& value,
                      const std::shared_ptr<DataType>& type);
};

struct ARROW_EXPORT StructScalar : public Scalar {
  using Scalar::Scalar;
  using TypeClass = StructType;
  using ValueType = std::vector<std::shared_ptr<Scalar>>;

  std::vector<std::shared_ptr<Scalar>> value;

  StructScalar(ValueType value, std::shared_ptr<DataType> type)
      : Scalar(std::move(type), true), value(std::move(value)) {}
};

class ARROW_EXPORT UnionScalar : public Scalar {
  using Scalar::Scalar;
  using TypeClass = UnionType;
};

class ARROW_EXPORT DictionaryScalar : public Scalar {
  using Scalar::Scalar;
  using TypeClass = DictionaryType;
};

class ARROW_EXPORT ExtensionScalar : public Scalar {
  using Scalar::Scalar;
  using TypeClass = ExtensionType;
};

ARROW_EXPORT
std::shared_ptr<Scalar> MakeNullScalar(const std::shared_ptr<DataType>& type);

namespace internal {

inline Status CheckBufferLength(...) { return Status::OK(); }

ARROW_EXPORT Status CheckBufferLength(const FixedSizeBinaryType* t,
                                      const std::shared_ptr<Buffer>* b);

template <typename T, typename Enable = void>
struct is_simple_scalar : std::false_type {};

template <typename T>
struct is_simple_scalar<
    T,
    typename std::enable_if<
        // scalar has a single extra data member named "value" with type "ValueType"
        std::is_same<decltype(std::declval<T>().value), typename T::ValueType>::value &&
        // scalar is constructible from (value, type)
        std::is_constructible<T, typename T::ValueType,
                              std::shared_ptr<DataType>>::value>::type> : std::true_type {
};

};  // namespace internal

template <typename ValueRef>
struct MakeScalarImpl {
  template <
      typename T, typename ScalarType = typename TypeTraits<T>::ScalarType,
      typename ValueType = typename ScalarType::ValueType,
      typename Enable = typename std::enable_if<
          internal::is_simple_scalar<ScalarType>::value &&
          std::is_same<ValueType, typename std::decay<ValueRef>::type>::value>::type>
  Status Visit(const T& t) {
    ARROW_RETURN_NOT_OK(internal::CheckBufferLength(&t, &value_));
    *out_ = std::make_shared<ScalarType>(ValueType(static_cast<ValueRef>(value_)), type_);
    return Status::OK();
  }

  Status Visit(const DataType& t) {
    return Status::NotImplemented("constructing scalars of type ", t, " from ", value_);
  }

  const std::shared_ptr<DataType>& type_;
  ValueRef value_;
  std::shared_ptr<Scalar>* out_;
};

template <typename Value>
Result<std::shared_ptr<Scalar>> MakeScalar(const std::shared_ptr<DataType>& type,
                                           Value&& value) {
  std::shared_ptr<Scalar> out;
  MakeScalarImpl<Value&&> impl = {type, std::forward<Value>(value), &out};
  ARROW_RETURN_NOT_OK(VisitTypeInline(*type, &impl));
  return out;
}

/// \brief type inferring scalar factory
template <typename Value, typename Traits = CTypeTraits<typename std::decay<Value>::type>,
          typename ScalarType = typename Traits::ScalarType,
          typename Enable = decltype(ScalarType(std::declval<Value>(),
                                                Traits::type_singleton()))>
std::shared_ptr<Scalar> MakeScalar(Value value) {
  return std::make_shared<ScalarType>(std::move(value), Traits::type_singleton());
}

inline std::shared_ptr<Scalar> MakeScalar(std::string value) {
  return std::make_shared<StringScalar>(value);
}

}  // namespace arrow
