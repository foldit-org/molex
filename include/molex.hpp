// molex C++ API.
//
// Hand-written RAII wrappers on top of the cbindgen-generated C ABI in
// `molex.h`. The C++ side lives in `namespace molex` and exposes
// move-only owning handles plus non-owning views with range-for support.
//
// Example:
//   #include <molex.hpp>
//   auto a = molex::Assembly::from_pdb_str(pdb_str);
//   if (!a) { /* parser failed; see molex::last_error_message() */ }
//   for (auto entity : a->entities()) {
//     for (auto residue : entity.residues()) {
//       for (auto atom : residue.atoms()) {
//         // ...
//       }
//     }
//   }
//   auto bytes = a->to_pdb();
//
// The wrapper does not throw on parse / write failure; instead, parsers
// return `std::optional<Assembly>` and writers return
// `std::optional<std::vector<uint8_t>>`. Callers retrieve the error
// message via `molex::last_error_message()`.

#ifndef MOLEX_HPP
#define MOLEX_HPP

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "molex.h"

namespace molex {

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

class Assembly;
class Entity;
class Residue;
class Atom;
class EditList;

template <typename T> class IndexedView;

// Re-export C-side payload types so callers can spell them as
// `molex::AtomRow`, `molex::Variant`, etc.
using AtomRow = ::molex_AtomRow;
using Variant = ::molex_Variant;
using VariantKind = ::molex_VariantKind;
using ProtonationKind = ::molex_ProtonationKind;
using EditKind = ::molex_EditKind;

// ---------------------------------------------------------------------------
// MoleculeType (alias to the C enum so callers write `molex::MoleculeType`)
// ---------------------------------------------------------------------------

using MoleculeType = ::molex_MoleculeType;

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Most recent error message recorded on this thread, or empty when
/// no failure has been observed since the last successful call.
inline std::string last_error_message() {
  std::size_t len = 0;
  const char* ptr = ::molex_last_error_message(&len);
  if (ptr == nullptr || len == 0) {
    return {};
  }
  return std::string(ptr, len);
}

// ---------------------------------------------------------------------------
// IndexedView: range-for adapter over a count + index-keyed getter
// ---------------------------------------------------------------------------

template <typename T>
class IndexedView {
 public:
  using Getter = std::function<T(std::size_t)>;

  IndexedView(std::size_t count, Getter getter)
      : count_(count), getter_(std::move(getter)) {}

  std::size_t size() const noexcept { return count_; }
  bool empty() const noexcept { return count_ == 0; }
  T operator[](std::size_t i) const { return getter_(i); }

  class Iterator {
   public:
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    Iterator(const IndexedView* view, std::size_t idx)
        : view_(view), idx_(idx) {}

    T operator*() const { return (*view_)[idx_]; }
    Iterator& operator++() {
      ++idx_;
      return *this;
    }
    bool operator==(const Iterator& other) const {
      return view_ == other.view_ && idx_ == other.idx_;
    }
    bool operator!=(const Iterator& other) const { return !(*this == other); }

   private:
    const IndexedView* view_;
    std::size_t idx_;
  };

  Iterator begin() const { return Iterator{this, 0}; }
  Iterator end() const { return Iterator{this, count_}; }

 private:
  std::size_t count_;
  Getter getter_;
};

// ---------------------------------------------------------------------------
// Atom: non-owning view of one atom inside an entity
// ---------------------------------------------------------------------------

class Atom {
 public:
  explicit Atom(const ::molex_Atom* handle) : handle_(handle) {}

  const ::molex_Atom* handle() const noexcept { return handle_; }

  /// Raw 4-byte PDB-style atom name (space-padded).
  std::string_view raw_name() const noexcept {
    std::size_t len = 0;
    const std::uint8_t* ptr = ::molex_atom_name(handle_, &len);
    if (ptr == nullptr || len == 0) {
      return {};
    }
    return std::string_view(reinterpret_cast<const char*>(ptr), len);
  }

  /// Logical atom name with leading/trailing ASCII spaces stripped.
  std::string_view name() const noexcept {
    auto raw = raw_name();
    while (!raw.empty() && raw.back() == ' ') {
      raw.remove_suffix(1);
    }
    while (!raw.empty() && raw.front() == ' ') {
      raw.remove_prefix(1);
    }
    return raw;
  }

  std::uint8_t atomic_number() const noexcept {
    return ::molex_atom_atomic_number(handle_);
  }

  struct Position {
    float x;
    float y;
    float z;
  };

  Position position() const noexcept {
    float xyz[3] = {0.0F, 0.0F, 0.0F};
    ::molex_atom_position(handle_, xyz);
    return Position{xyz[0], xyz[1], xyz[2]};
  }

  float occupancy() const noexcept {
    return ::molex_atom_occupancy(handle_);
  }
  float b_factor() const noexcept {
    return ::molex_atom_b_factor(handle_);
  }
  std::int8_t formal_charge() const noexcept {
    return ::molex_atom_formal_charge(handle_);
  }

 private:
  const ::molex_Atom* handle_;
};

// ---------------------------------------------------------------------------
// Residue: non-owning view of one polymer residue inside an entity
// ---------------------------------------------------------------------------

class Residue {
 public:
  Residue(const ::molex_Entity* entity, const ::molex_Residue* handle,
          std::size_t index)
      : entity_(entity), handle_(handle), index_(index) {}

  const ::molex_Residue* handle() const noexcept { return handle_; }

  /// 3-byte residue name (e.g. "ALA") with trailing ASCII spaces
  /// stripped.
  std::string_view name() const noexcept {
    std::size_t len = 0;
    const std::uint8_t* ptr = ::molex_residue_name(handle_, &len);
    if (ptr == nullptr || len == 0) {
      return {};
    }
    std::string_view raw(reinterpret_cast<const char*>(ptr), len);
    while (!raw.empty() && raw.back() == ' ') {
      raw.remove_suffix(1);
    }
    return raw;
  }

  /// Author-side residue sequence id (falls back to label_seq_id).
  std::int32_t seq_id() const noexcept {
    return ::molex_residue_seq_id(handle_);
  }

  std::int32_t label_seq_id() const noexcept {
    return ::molex_residue_label_seq_id(handle_);
  }

  std::uint8_t ins_code() const noexcept {
    return ::molex_residue_ins_code(handle_);
  }

  std::size_t num_atoms() const noexcept {
    return ::molex_entity_residue_num_atoms(entity_, index_);
  }

  Atom atom(std::size_t i) const noexcept {
    return Atom{::molex_entity_residue_atom(entity_, index_, i)};
  }

  /// Range-for-friendly view over the atoms of this residue.
  IndexedView<Atom> atoms() const noexcept {
    const ::molex_Entity* entity = entity_;
    std::size_t index = index_;
    return IndexedView<Atom>{
        num_atoms(), [entity, index](std::size_t i) -> Atom {
          return Atom{::molex_entity_residue_atom(entity, index, i)};
        }};
  }

 private:
  const ::molex_Entity* entity_;
  const ::molex_Residue* handle_;
  std::size_t index_;
};

// ---------------------------------------------------------------------------
// Entity: non-owning view of one entity inside an assembly
// ---------------------------------------------------------------------------

class Entity {
 public:
  explicit Entity(const ::molex_Entity* handle) : handle_(handle) {}

  const ::molex_Entity* handle() const noexcept { return handle_; }

  std::uint32_t id() const noexcept { return ::molex_entity_id(handle_); }

  MoleculeType molecule_type() const noexcept {
    return ::molex_entity_molecule_type(handle_);
  }

  /// PDB chain identifier byte for polymer entities; -1 otherwise.
  std::int32_t pdb_chain_id() const noexcept {
    return ::molex_entity_pdb_chain_id(handle_);
  }

  std::size_t num_atoms() const noexcept {
    return ::molex_entity_num_atoms(handle_);
  }

  Atom atom(std::size_t i) const noexcept {
    return Atom{::molex_entity_atom(handle_, i)};
  }

  IndexedView<Atom> atoms() const noexcept {
    const ::molex_Entity* entity = handle_;
    return IndexedView<Atom>{
        num_atoms(), [entity](std::size_t i) -> Atom {
          return Atom{::molex_entity_atom(entity, i)};
        }};
  }

  /// Number of indexable residues (0 for small-molecule / bulk).
  std::size_t num_residues() const noexcept {
    return ::molex_entity_num_residues(handle_);
  }

  Residue residue(std::size_t i) const noexcept {
    return Residue{handle_, ::molex_entity_residue(handle_, i), i};
  }

  IndexedView<Residue> residues() const noexcept {
    const ::molex_Entity* entity = handle_;
    return IndexedView<Residue>{
        num_residues(), [entity](std::size_t i) -> Residue {
          return Residue{entity, ::molex_entity_residue(entity, i), i};
        }};
  }

  /// Single 3-byte residue name for non-polymer entities
  /// (`SmallMolecule` / `Bulk`); empty string for polymers. Padded with
  /// ASCII spaces; trim if needed.
  std::string_view residue_name_single() const noexcept {
    std::size_t len = 0;
    const std::uint8_t* ptr =
        ::molex_entity_residue_name_single(handle_, &len);
    if (ptr == nullptr || len == 0) {
      return std::string_view{};
    }
    return std::string_view{reinterpret_cast<const char*>(ptr), len};
  }

  /// Number of equal-sized molecule chunks the atom set should be split
  /// into. 1 for `SmallMolecule`, `BulkEntity::molecule_count` for
  /// `Bulk`, 0 for polymers.
  std::size_t molecule_count() const noexcept {
    return ::molex_entity_molecule_count(handle_);
  }

 private:
  const ::molex_Entity* handle_;
};

// ---------------------------------------------------------------------------
// Assembly: owning RAII handle. Move-only.
// ---------------------------------------------------------------------------

class Assembly {
 public:
  Assembly() = default;
  Assembly(const Assembly&) = delete;
  Assembly& operator=(const Assembly&) = delete;

  Assembly(Assembly&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }
  Assembly& operator=(Assembly&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  ~Assembly() { reset(); }

  /// Parse a PDB-format string. Returns an empty optional on failure;
  /// retrieve the error via `molex::last_error_message()`.
  static std::optional<Assembly> from_pdb_str(std::string_view pdb) {
    ::molex_Assembly* h =
        ::molex_pdb_str_to_assembly(pdb.data(), pdb.size());
    if (h == nullptr) {
      return std::nullopt;
    }
    return Assembly{h};
  }

  /// Parse an mmCIF-format string. Returns an empty optional on failure.
  static std::optional<Assembly> from_cif_str(std::string_view cif) {
    ::molex_Assembly* h =
        ::molex_cif_str_to_assembly(cif.data(), cif.size());
    if (h == nullptr) {
      return std::nullopt;
    }
    return Assembly{h};
  }

  /// Decode BinaryCIF bytes. Returns an empty optional on failure.
  static std::optional<Assembly> from_bcif(
      const std::uint8_t* bytes, std::size_t len) {
    ::molex_Assembly* h = ::molex_bcif_to_assembly(bytes, len);
    if (h == nullptr) {
      return std::nullopt;
    }
    return Assembly{h};
  }

  static std::optional<Assembly> from_bcif(
      const std::vector<std::uint8_t>& bytes) {
    return from_bcif(bytes.data(), bytes.size());
  }

  /// Decode assembly binary bytes. Returns an empty optional on failure.
  static std::optional<Assembly> from_assembly_bytes(
      const std::uint8_t* bytes, std::size_t len) {
    ::molex_Assembly* h = ::molex_bytes_to_assembly(bytes, len);
    if (h == nullptr) {
      return std::nullopt;
    }
    return Assembly{h};
  }

  static std::optional<Assembly> from_assembly_bytes(
      const std::vector<std::uint8_t>& bytes) {
    return from_assembly_bytes(bytes.data(), bytes.size());
  }

  /// Emit this assembly as binary bytes. Empty optional on failure.
  std::optional<std::vector<std::uint8_t>> to_assembly_bytes() const {
    std::uint8_t* buf = nullptr;
    std::size_t len = 0;
    if (::molex_assembly_to_bytes(handle_, &buf, &len) != MOLEX_OK) {
      return std::nullopt;
    }
    std::vector<std::uint8_t> out(buf, buf + len);
    ::molex_free_bytes(buf, len);
    return out;
  }

  /// Emit this assembly as a PDB-format byte buffer. Empty optional on
  /// failure.
  std::optional<std::vector<std::uint8_t>> to_pdb() const {
    std::uint8_t* buf = nullptr;
    std::size_t len = 0;
    if (::molex_assembly_to_pdb(handle_, &buf, &len) != MOLEX_OK) {
      return std::nullopt;
    }
    std::vector<std::uint8_t> out(buf, buf + len);
    ::molex_free_bytes(buf, len);
    return out;
  }

  /// Apply every edit in `edits` to this assembly in order. Returns
  /// `false` on the first failure; edits applied before the failing
  /// one stay applied. See `molex::last_error_message()` for details.
  bool apply_edits(const EditList& edits);

  /// Decode delta bytes and apply them in one call. Equivalent to
  /// `EditList::from_delta` + `apply_edits` + drop, but avoids the
  /// intermediate handle on the apply hot path.
  bool apply_delta(const std::uint8_t* bytes, std::size_t len) {
    return ::molex_assembly_apply_delta(handle_, bytes, len) == MOLEX_OK;
  }

  bool apply_delta(const std::vector<std::uint8_t>& bytes) {
    return apply_delta(bytes.data(), bytes.size());
  }

  std::uint64_t generation() const noexcept {
    return ::molex_assembly_generation(handle_);
  }

  std::size_t num_entities() const noexcept {
    return ::molex_assembly_num_entities(handle_);
  }

  Entity entity(std::size_t i) const noexcept {
    return Entity{::molex_assembly_entity(handle_, i)};
  }

  IndexedView<Entity> entities() const noexcept {
    const ::molex_Assembly* a = handle_;
    return IndexedView<Entity>{
        num_entities(), [a](std::size_t i) -> Entity {
          return Entity{::molex_assembly_entity(a, i)};
        }};
  }

  const ::molex_Assembly* handle() const noexcept { return handle_; }

 private:
  explicit Assembly(::molex_Assembly* h) : handle_(h) {}

  void reset() noexcept {
    if (handle_ != nullptr) {
      ::molex_assembly_free(handle_);
      handle_ = nullptr;
    }
  }

  ::molex_Assembly* handle_ = nullptr;
};

// ---------------------------------------------------------------------------
// EditList: owning RAII handle over `molex_EditList`. Move-only.
// ---------------------------------------------------------------------------

class EditList {
 public:
  EditList() : handle_(::molex_edits_new()) {}

  ~EditList() { reset(); }

  EditList(const EditList&) = delete;
  EditList& operator=(const EditList&) = delete;

  EditList(EditList&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }
  EditList& operator=(EditList&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  /// Number of edits currently in this list.
  std::size_t count() const noexcept {
    return ::molex_edits_count(handle_);
  }

  /// Append a `SetEntityCoords` edit. Returns `false` on failure.
  bool push_set_entity_coords(
      std::uint32_t entity_id,
      const float* coords_xyz,
      std::size_t coord_count) {
    return ::molex_edits_push_set_entity_coords(
               handle_, entity_id, coords_xyz, coord_count) == MOLEX_OK;
  }

  /// Append a `SetResidueCoords` edit. Returns `false` on failure.
  bool push_set_residue_coords(
      std::uint32_t entity_id,
      std::size_t residue_idx,
      const float* coords_xyz,
      std::size_t coord_count) {
    return ::molex_edits_push_set_residue_coords(
               handle_, entity_id, residue_idx, coords_xyz, coord_count)
        == MOLEX_OK;
  }

  /// Append a `MutateResidue` edit. `new_name` must point to 3 bytes;
  /// `atoms` and `variants` are borrowed for the duration of the call.
  bool push_mutate_residue(
      std::uint32_t entity_id,
      std::size_t residue_idx,
      const std::uint8_t* new_name_3,
      const AtomRow* atoms,
      std::size_t atom_count,
      const Variant* variants,
      std::size_t variant_count) {
    return ::molex_edits_push_mutate_residue(
               handle_, entity_id, residue_idx, new_name_3, atoms,
               atom_count, variants, variant_count) == MOLEX_OK;
  }

  /// Append a `SetVariants` edit.
  bool push_set_variants(
      std::uint32_t entity_id,
      std::size_t residue_idx,
      const Variant* variants,
      std::size_t variant_count) {
    return ::molex_edits_push_set_variants(
               handle_, entity_id, residue_idx, variants, variant_count)
        == MOLEX_OK;
  }

  /// Serialize this list as delta bytes. Empty optional on failure
  /// (e.g., when the list contains a topology edit). See
  /// `molex::last_error_message()` for details.
  std::optional<std::vector<std::uint8_t>> to_delta() const {
    std::uint8_t* buf = nullptr;
    std::size_t len = 0;
    if (::molex_edits_to_delta(handle_, &buf, &len) != MOLEX_OK) {
      return std::nullopt;
    }
    std::vector<std::uint8_t> out(buf, buf + len);
    ::molex_free_bytes(buf, len);
    return out;
  }

  /// Decode delta bytes into a new edit list. Empty optional on
  /// failure.
  static std::optional<EditList> from_delta(
      const std::uint8_t* bytes, std::size_t len) {
    ::molex_EditList* h = ::molex_delta_to_edits(bytes, len);
    if (h == nullptr) {
      return std::nullopt;
    }
    return EditList{h};
  }

  static std::optional<EditList> from_delta(
      const std::vector<std::uint8_t>& bytes) {
    return from_delta(bytes.data(), bytes.size());
  }

  const ::molex_EditList* handle() const noexcept { return handle_; }

  // -------------------------------------------------------------------------
  // Read accessors (per-variant views)
  // -------------------------------------------------------------------------
  //
  // The C API exposes per-kind getters that return either borrowed
  // pointers into the EditList (coords, residue name) or freshly
  // heap-allocated temporary arrays (atoms, variants) that the caller
  // must free via `molex_atom_rows_free` / `molex_variants_free`. The
  // RAII wrappers below encapsulate that lifecycle so consumers never
  // call the free functions directly.
  //
  // All views borrow from the parent EditList; they MUST NOT outlive
  // the EditList or be used after the list is mutated (push_*,
  // assignment, etc.).

  /// View of a `SetEntityCoords` edit. Coords are flat `[x, y, z, ...]`.
  class SetEntityCoordsView {
   public:
    std::uint32_t entity_id{};
    const float* coords_xyz{nullptr};
    std::size_t coord_count{0};  // number of (x,y,z) triples
  };

  /// View of a `SetResidueCoords` edit.
  class SetResidueCoordsView {
   public:
    std::uint32_t entity_id{};
    std::size_t residue_idx{0};
    const float* coords_xyz{nullptr};
    std::size_t coord_count{0};
  };

  /// View of a `MutateResidue` edit. Owns the temporary atom + variant
  /// arrays returned by the C accessor; frees them on destruction.
  /// Move-only.
  class MutateResidueView {
   public:
    std::uint32_t entity_id{};
    std::size_t residue_idx{0};
    /// 3 bytes, space-padded; borrowed from the parent EditList.
    const std::uint8_t* new_name{nullptr};
    const AtomRow* atoms{nullptr};
    std::size_t atom_count{0};
    const Variant* variants{nullptr};
    std::size_t variant_count{0};

    MutateResidueView() = default;
    ~MutateResidueView() { free_temps(); }

    MutateResidueView(const MutateResidueView&) = delete;
    MutateResidueView& operator=(const MutateResidueView&) = delete;

    MutateResidueView(MutateResidueView&& other) noexcept { swap_from(other); }
    MutateResidueView& operator=(MutateResidueView&& other) noexcept {
      if (this != &other) {
        free_temps();
        swap_from(other);
      }
      return *this;
    }

   private:
    void swap_from(MutateResidueView& other) noexcept {
      entity_id = other.entity_id;
      residue_idx = other.residue_idx;
      new_name = other.new_name;
      atoms = other.atoms;
      atom_count = other.atom_count;
      variants = other.variants;
      variant_count = other.variant_count;
      other.atoms = nullptr;
      other.atom_count = 0;
      other.variants = nullptr;
      other.variant_count = 0;
    }
    void free_temps() noexcept {
      if (atoms != nullptr) {
        ::molex_atom_rows_free(
            const_cast<AtomRow*>(atoms), atom_count);
        atoms = nullptr;
        atom_count = 0;
      }
      if (variants != nullptr) {
        ::molex_variants_free(
            const_cast<Variant*>(variants), variant_count);
        variants = nullptr;
        variant_count = 0;
      }
    }
  };

  /// View of a `SetVariants` edit. Owns the temporary variant array.
  /// Move-only.
  class SetVariantsView {
   public:
    std::uint32_t entity_id{};
    std::size_t residue_idx{0};
    const Variant* variants{nullptr};
    std::size_t variant_count{0};

    SetVariantsView() = default;
    ~SetVariantsView() { free_temps(); }

    SetVariantsView(const SetVariantsView&) = delete;
    SetVariantsView& operator=(const SetVariantsView&) = delete;

    SetVariantsView(SetVariantsView&& other) noexcept { swap_from(other); }
    SetVariantsView& operator=(SetVariantsView&& other) noexcept {
      if (this != &other) {
        free_temps();
        swap_from(other);
      }
      return *this;
    }

   private:
    void swap_from(SetVariantsView& other) noexcept {
      entity_id = other.entity_id;
      residue_idx = other.residue_idx;
      variants = other.variants;
      variant_count = other.variant_count;
      other.variants = nullptr;
      other.variant_count = 0;
    }
    void free_temps() noexcept {
      if (variants != nullptr) {
        ::molex_variants_free(
            const_cast<Variant*>(variants), variant_count);
        variants = nullptr;
        variant_count = 0;
      }
    }
  };

  /// Tagged union of all readable edit kinds (the four steady-state
  /// variants that ride the delta wire -- `AddEntity` / `RemoveEntity`
  /// are not exposed because they don't survive the wire round trip).
  /// `EditList::view_at` returns this; consumers `std::visit` over it.
  using EditView = std::variant<
      SetEntityCoordsView,
      SetResidueCoordsView,
      MutateResidueView,
      SetVariantsView>;

  /// Kind of the edit at `index`, or `EditKind::Invalid` if the index
  /// is out of range.
  EditKind kind_at(std::size_t index) const noexcept {
    return ::molex_edits_kind_at(handle_, index);
  }

  /// Read the edit at `index` as a typed view. Empty optional when
  /// the index is out of range or the edit is a topology edit
  /// (`AddEntity` / `RemoveEntity`); see `kind_at` to distinguish.
  ///
  /// (Variant constants below are the cbindgen-emitted names; the
  /// project's cbindgen config renames Rust PascalCase variants to
  /// `ScreamingSnakeCase` with the enum-name prefix.)
  std::optional<EditView> view_at(std::size_t index) const {
    switch (::molex_edits_kind_at(handle_, index)) {
      case MOLEX_EDIT_KIND_SET_ENTITY_COORDS: {
        SetEntityCoordsView v;
        if (::molex_edits_set_entity_coords_at(
                handle_, index, &v.entity_id, &v.coords_xyz,
                &v.coord_count) != MOLEX_OK) {
          return std::nullopt;
        }
        return EditView{std::move(v)};
      }
      case MOLEX_EDIT_KIND_SET_RESIDUE_COORDS: {
        SetResidueCoordsView v;
        if (::molex_edits_set_residue_coords_at(
                handle_, index, &v.entity_id, &v.residue_idx,
                &v.coords_xyz, &v.coord_count) != MOLEX_OK) {
          return std::nullopt;
        }
        return EditView{std::move(v)};
      }
      case MOLEX_EDIT_KIND_MUTATE_RESIDUE: {
        MutateResidueView v;
        if (::molex_edits_mutate_residue_at(
                handle_, index, &v.entity_id, &v.residue_idx,
                &v.new_name, &v.atoms, &v.atom_count, &v.variants,
                &v.variant_count) != MOLEX_OK) {
          return std::nullopt;
        }
        return EditView{std::move(v)};
      }
      case MOLEX_EDIT_KIND_SET_VARIANTS: {
        SetVariantsView v;
        if (::molex_edits_set_variants_at(
                handle_, index, &v.entity_id, &v.residue_idx,
                &v.variants, &v.variant_count) != MOLEX_OK) {
          return std::nullopt;
        }
        return EditView{std::move(v)};
      }
      case MOLEX_EDIT_KIND_INVALID:
      case MOLEX_EDIT_KIND_ADD_ENTITY:
      case MOLEX_EDIT_KIND_REMOVE_ENTITY:
      default:
        return std::nullopt;
    }
  }

 private:
  explicit EditList(::molex_EditList* h) : handle_(h) {}

  void reset() noexcept {
    if (handle_ != nullptr) {
      ::molex_edits_free(handle_);
      handle_ = nullptr;
    }
  }

  ::molex_EditList* handle_ = nullptr;
};

// Assembly::apply_edits — out-of-class since it depends on the full
// EditList declaration.
inline bool Assembly::apply_edits(const EditList& edits) {
  return ::molex_assembly_apply_edits(handle_, edits.handle()) == MOLEX_OK;
}

}  // namespace molex

#endif  // MOLEX_HPP
