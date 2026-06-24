# Type stubs for the `molex` extension module.
#
# HAND-MAINTAINED. There is no auto-generation step: when the pyo3 surface in
# `src/python/` (mod.rs, entity.rs, io.rs, arrays.rs, views.rs) or the module
# registration in `src/lib.rs` changes, update this file by hand to match. The
# registered surface is the `#[pyclass]` / `#[pymethods]` / `#[pyfunction]`
# items wired in the `molex` `#[pymodule]` (`src/lib.rs`).
#
# Layout note: this is a pure-Rust extension (no `python-source`). maturin
# ships this file from the project root alongside an auto-generated `py.typed`
# marker (see pyproject.toml `[tool.maturin]`).
#
# Mapping conventions used here:
#   Rust `Vec<u8>`        -> Python `bytes`
#   Rust `String`/`&str`  -> Python `str`
#   Rust `Option<String>` -> Python `str | None`
#   Rust `u32/u64/usize/i32` -> Python `int`
#   Rust `f32`            -> Python `float`
#   numpy columns         -> `Any` (numpy is not a typing dependency here)
#   Biotite AtomArray     -> `Any`

import enum
from typing import Any

# ---------------------------------------------------------------------------
# Object API (primary surface)
#
# The documented default for a normal caller: parse a structure with
# `Assembly.from_pdb` / `from_mmcif` / `from_bcif`, walk `entities()` ->
# `Entity` -> `residues()` -> `Residue`, and read atoms as numpy columns via
# `to_arrays()`. Completion (missing heavy atoms filled) runs at parse time.
# ---------------------------------------------------------------------------

class Completion(enum.Enum):
    """Completion level for parsing and re-completion: `Raw < Heavy < AllAtom`.

    `Raw` keeps atoms exactly as given (no fabricated heavy atoms, input
    hydrogens preserved); `Heavy` fabricates absent heavy template atoms and
    drops input hydrogens (the file-ingest default); `AllAtom` additionally
    places template hydrogens."""

    Raw = ...
    Heavy = ...
    AllAtom = ...

class Assembly:
    """Object-graph handle over a parsed molecular assembly.

    Primary entry points are the `from_pdb` / `from_mmcif` / `from_bcif`
    static methods; navigate via `entities()` and read atoms via `to_arrays()`.
    """

    @staticmethod
    def from_pdb(
        text: str, completion: Completion = Completion.Heavy
    ) -> Assembly:
        """Parse a PDB string at the given `completion` level (default
        `Completion.Heavy`: missing heavy atoms filled, input hydrogens
        dropped). Secondary structure is left empty (call `recompute_ss()`)."""
        ...
    @staticmethod
    def from_mmcif(
        text: str, completion: Completion = Completion.Heavy
    ) -> Assembly:
        """Parse an mmCIF string (see `from_pdb` for `completion`)."""
        ...
    @staticmethod
    def from_bcif(
        bytes: bytes, completion: Completion = Completion.Heavy
    ) -> Assembly:
        """Parse BinaryCIF bytes (see `from_pdb` for `completion`)."""
        ...
    @staticmethod
    def from_assembly_bytes(bytes: bytes) -> Assembly:
        """Decode assembly wire bytes into an `Assembly`."""
        ...
    def to_assembly_bytes(self) -> bytes:
        """Emit the assembly as wire bytes."""
        ...
    def to_mmcif(self) -> str:
        """Emit the assembly as an mmCIF string (one `_atom_site` loop). The
        honest write path for assemblies the legacy PDB writer refuses
        (multi-character chains, >62 chains): every chain id is emitted verbatim
        into `label_asym_id` and the output round-trips through `from_mmcif`."""
        ...
    @property
    def generation(self) -> int:
        """Monotonic generation counter."""
        ...
    def complete(self, level: Completion) -> Assembly:
        """Fresh assembly re-completed at `level`: polymer entities are rebuilt
        (Raw keeps atoms as-is, Heavy fills heavy atoms, AllAtom adds template
        hydrogens). Atom indices shift; externally held indices do not carry
        over."""
        ...
    def normalize(self) -> Assembly:
        """Fresh heavy-complete assembly: polymer entities gain missing heavy
        atoms. Equivalent to `complete(Completion.Heavy)`."""
        ...
    def to_all_atom(self) -> Assembly:
        """Fresh all-atom assembly: polymer entities gain template hydrogens.
        Equivalent to `complete(Completion.AllAtom)`."""
        ...
    def apply_edits(self, edits: EditList) -> None:
        """Apply every edit in `edits` in order, advancing `generation`."""
        ...
    def apply_delta(self, bytes: bytes) -> None:
        """Decode delta bytes and apply them in one call."""
        ...
    def recompute_ss(self) -> None:
        """Recompute per-entity secondary structure in place (opt-in DSSP)."""
        ...
    def secondary_structure(self) -> list[str]:
        """Per-entity Q3 secondary structure, one string per entity aligned to
        `entities()` order. Each protein entity's string has one char per
        residue: `'H'` (helix), `'E'` (sheet), `'C'` (coil); non-protein
        entities get `""`. Reads the SS from `recompute_ss`; call that first,
        otherwise every protein string is all-`'C'`."""
        ...
    def sasa(self, probe_radius: float = 1.4) -> float:
        """Total Shrake-Rupley solvent-accessible surface area (Angstrom^2)
        over protein atoms. Water, ions, ligands, and nucleic acids are
        excluded (freesasa/biotite protein-scope convention). Uses Bondi van
        der Waals radii and ~960 test points per atom; `probe_radius` is the
        solvent radius in Angstroms (1.4 for water)."""
        ...
    def entities(self) -> list[Entity]:
        """The assembly's entities, in order."""
        ...
    def to_arrays(self) -> AtomArrays:
        """Every atom as per-atom numpy columns (Biotite-free)."""
        ...
    def covalent_bonds(self) -> list[tuple[int, int]]:
        """Distance-based covalent bonds over the whole assembly, as atom-index
        pairs `(i, j)` (with `i < j`) in `to_arrays()` flat order. Same
        distance/covalent-radius perception molex uses for ligands, run across
        every atom, so it bonds across residue and entity boundaries.
        Hydrogen-to-hydrogen contacts are not bonded."""
        ...
    def disulfides(self) -> list[tuple[int, int]]:
        """Cys SG-SG disulfide bonds within the disulfide distance band, intra-
        and inter-chain alike, as atom-index pairs in `to_arrays()` flat
        order."""
        ...
    def __len__(self) -> int:
        """Number of entities in the assembly."""
        ...
    def __getitem__(self, index: int) -> Entity:
        """The entity at `index` (negative indexes from the end). With
        `__len__`, makes `for e in assembly:` iterate. Raises `IndexError`
        when out of range."""
        ...
    def __repr__(self) -> str:
        """`<Assembly: {n} entities, gen {g}>`."""
        ...

class Entity:
    """One entity in an `Assembly` (a cheap `Arc` refcount handle).

    Reached from `Assembly.entities()`; all members are read accessors."""

    @property
    def id(self) -> int:
        """Raw entity id."""
        ...
    @property
    def kind(self) -> str:
        """Structural discriminant: `"Protein"`, `"NucleicAcid"`,
        `"SmallMolecule"`, or `"Bulk"`."""
        ...
    @property
    def molecule_type(self) -> str:
        """Molecule-type classification, e.g. `"Protein"`, `"Ligand"`,
        `"Ion"`, `"Water"`."""
        ...
    @property
    def chain_id(self) -> str | None:
        """PDB chain id (the mmCIF `label_asym_id`, which may be multi-character
        for large assemblies such as ribosomes and capsids), or `None` when the
        entity has no chain id (a non-polymer entity)."""
        ...
    @property
    def label(self) -> str:
        """Human-readable label, e.g. `"Protein Chain A"`, `"Ligand (ATP)"`."""
        ...
    @property
    def atom_count(self) -> int:
        """Total atom count in this entity."""
        ...
    @property
    def residue_count(self) -> int:
        """Residue count; 0 for non-polymer entities."""
        ...
    def residues(self) -> list[Residue]:
        """The entity's residues, in order. Empty for a non-polymer entity."""
        ...
    def phi_psi(self) -> list[tuple[float | None, float | None]]:
        """Backbone (phi, psi) torsion angles per residue, in degrees, in
        residue order (aligned 1:1 with `residues()`). Each element is `None`
        at a chain terminus or break, or when the residue (or its neighbour)
        lacks a complete N/CA/C backbone: `phi` is `None` for the first
        residue, `psi` for the last. Angles are in `(-180, 180]`. Empty list
        for a non-protein entity."""
        ...
    def to_arrays(self) -> AtomArrays:
        """This entity's atoms as per-atom numpy columns (Biotite-free)."""
        ...
    def __len__(self) -> int:
        """Residue count; mirrors `len(assembly)` so `len(entity)` works too."""
        ...
    def __getitem__(self, index: int) -> Residue:
        """The residue at `index` (negative indexes from the end). With
        `__len__`, makes `for r in entity:` iterate. Raises `IndexError` for
        an out-of-range index or a non-polymer entity (no residues)."""
        ...
    def __repr__(self) -> str:
        """`<Entity {id} {kind} chain={chain_id} {atom_count} atoms>`."""
        ...

class Residue:
    """One residue in a polymer entity (a cheap `Arc` refcount handle into the
    parent entity plus a residue index).

    Reached from `Entity.residues()` or `entity[i]`."""

    @property
    def name(self) -> str:
        """Trimmed 3-letter residue name, e.g. `"ALA"`."""
        ...
    @property
    def seq_id(self) -> int:
        """Author-side sequence id (falls back to `label_seq_id` when absent)."""
        ...
    @property
    def label_seq_id(self) -> int:
        """Structural-side sequence id."""
        ...
    @property
    def ins_code(self) -> str | None:
        """Insertion code as a one-char string, or `None` when blank."""
        ...
    def to_arrays(self) -> AtomArrays:
        """This residue's atoms as per-atom numpy columns (Biotite-free)."""
        ...
    def __repr__(self) -> str:
        """`<Residue {name} {seq_id}{ins_code}>`."""
        ...

class AtomArrays:
    """Per-atom annotation columns as numpy arrays (the native, Biotite-free
    read path). Each attribute is a `numpy.ndarray`; bonds are not exposed
    here (a Biotite-only concern)."""

    coords: Any
    """`(n, 3)` float32 coordinates."""
    atom_names: Any
    """Per-atom PDB atom names."""
    elements: Any
    """Per-atom element symbols."""
    res_ids: Any
    """Per-atom residue ids (int32)."""
    res_names: Any
    """Per-atom residue names."""
    chain_ids: Any
    """Per-atom chain ids."""
    occupancies: Any
    """Per-atom occupancies (float32)."""
    b_factors: Any
    """Per-atom B-factors (float32)."""
    entity_ids: Any
    """Per-atom AtomWorks entity ids (int32)."""
    mol_types: Any
    """Per-atom AtomWorks `mol_type` strings."""
    chain_types: Any
    """Per-atom AtomWorks `chain_type` codes (int32)."""
    def __len__(self) -> int:
        """Number of atoms (read from the `(n, 3)` `coords` anchor)."""
        ...
    def __repr__(self) -> str:
        """`<AtomArrays: {n} atoms>`."""
        ...

# ---------------------------------------------------------------------------
# Edit / delta handle surface
#
# Build a typed list of assembly edits, serialize to delta bytes, or read
# edits back out as the `*View` value classes. Parallels the C API in
# `c_api::edit`.
# ---------------------------------------------------------------------------

class EditList:
    """An ordered list of typed assembly edits."""

    def __init__(self) -> None:
        """Construct an empty edit list."""
        ...
    def __len__(self) -> int:
        """Number of edits currently in the list."""
        ...
    def push_set_entity_coords(
        self, entity_id: int, coords: list[tuple[float, float, float]]
    ) -> None:
        """Append a `SetEntityCoords` edit."""
        ...
    def push_set_residue_coords(
        self,
        entity_id: int,
        residue_idx: int,
        coords: list[tuple[float, float, float]],
    ) -> None:
        """Append a `SetResidueCoords` edit."""
        ...
    def push_mutate_residue(
        self,
        entity_id: int,
        residue_idx: int,
        new_name: bytes,
        atoms: list[AtomRow],
        variants: list[Variant],
    ) -> None:
        """Append a `MutateResidue` edit. `new_name` is a 3-byte residue
        name (e.g. `b"HIS"`). Note: the edit-side names here take raw bytes
        (`new_name`, `AtomRow.name`), unlike `Residue.name`, which is a
        decoded `str`."""
        ...
    def push_set_variants(
        self, entity_id: int, residue_idx: int, variants: list[Variant]
    ) -> None:
        """Append a `SetVariants` edit."""
        ...
    def to_delta(self) -> bytes:
        """Serialize this list as delta bytes."""
        ...
    @staticmethod
    def from_delta(bytes: bytes) -> EditList:
        """Decode delta bytes into a new edit list."""
        ...
    def kind_at(self, index: int) -> int:
        """Kind of the edit at `index` (an `EditKind.*`-style integer:
        1=SetEntityCoords, 2=SetResidueCoords, 3=MutateResidue,
        4=SetVariants, 5=AddEntity, 6=RemoveEntity)."""
        ...
    def set_entity_coords_at(self, index: int) -> SetEntityCoordsView:
        """Read the `SetEntityCoords` edit at `index`."""
        ...
    def set_residue_coords_at(self, index: int) -> SetResidueCoordsView:
        """Read the `SetResidueCoords` edit at `index`."""
        ...
    def mutate_residue_at(self, index: int) -> MutateResidueView:
        """Read the `MutateResidue` edit at `index`."""
        ...
    def set_variants_at(self, index: int) -> SetVariantsView:
        """Read the `SetVariants` edit at `index`."""
        ...

class Variant:
    """A single residue variant tag (terminus / disulfide / protonation)."""

    @staticmethod
    def n_terminus() -> Variant:
        """Chain N-terminus patch."""
        ...
    @staticmethod
    def c_terminus() -> Variant:
        """Chain C-terminus patch."""
        ...
    @staticmethod
    def disulfide() -> Variant:
        """Participates in a disulfide bond."""
        ...
    @staticmethod
    def protonation_his_delta() -> Variant:
        """`HID` - delta-protonated histidine."""
        ...
    @staticmethod
    def protonation_his_epsilon() -> Variant:
        """`HIE` - epsilon-protonated histidine."""
        ...
    @staticmethod
    def protonation_his_doubly() -> Variant:
        """`HIP` - doubly-protonated histidine."""
        ...
    @staticmethod
    def protonation_custom(name: str) -> Variant:
        """Custom protonation state, e.g. `"ASP_PROTONATED"`."""
        ...
    @staticmethod
    def other(name: str) -> Variant:
        """Open-ended variant tag carrying an arbitrary string."""
        ...

class AtomRow:
    """A single atom row: position + name + element."""

    position_x: float
    position_y: float
    position_z: float
    name: bytes
    """PDB atom name (4 bytes, space-padded; e.g. `b"CA  "`)."""
    element: str
    """Element symbol (e.g. `"C"`, `"Fe"`)."""
    def __init__(
        self,
        position_x: float,
        position_y: float,
        position_z: float,
        name: bytes,
        element: str,
    ) -> None:
        """Construct an `AtomRow` from positional fields."""
        ...

class SetEntityCoordsView:
    """Read view of a `SetEntityCoords` edit (from
    `EditList.set_entity_coords_at`)."""

    entity_id: int
    coords: list[tuple[float, float, float]]

class SetResidueCoordsView:
    """Read view of a `SetResidueCoords` edit."""

    entity_id: int
    residue_idx: int
    coords: list[tuple[float, float, float]]

class MutateResidueView:
    """Read view of a `MutateResidue` edit."""

    entity_id: int
    residue_idx: int
    new_name: bytes
    """New 3-letter residue name (space-padded)."""
    atoms: list[AtomRow]
    variants: list[Variant]

class SetVariantsView:
    """Read view of a `SetVariants` edit."""

    entity_id: int
    residue_idx: int
    variants: list[Variant]

# ---------------------------------------------------------------------------
# Scalar analysis free functions
# ---------------------------------------------------------------------------

def rmsd(a: Any, b: Any) -> float:
    """Optimal-superposition RMSD between two `(N, 3)` coordinate sets. Each
    argument is an array-like of shape `(N, 3)`: a numpy array or a sequence of
    `(x, y, z)` tuples. Superposes `a` onto `b` with the Kabsch algorithm and
    returns the scalar RMSD, invariant to any rigid motion of either set. Raises
    `ValueError` if the sets differ in length or have fewer than three points."""
    ...

# ---------------------------------------------------------------------------
# Transport / serialization free functions
#
# Byte-blob helpers kept for the plugin wire protocol, not the front door.
# A normal caller wants the object API above.
# ---------------------------------------------------------------------------

def pdb_to_assembly_bytes(pdb_str: str) -> bytes:
    """Parse a PDB string and emit assembly wire bytes. Transport helper; prefer
    `Assembly.from_pdb`."""
    ...

def mmcif_to_assembly_bytes(cif_str: str) -> bytes:
    """Parse an mmCIF string and emit assembly wire bytes. Transport helper; prefer
    `Assembly.from_mmcif`."""
    ...

def bcif_to_assembly_bytes(bytes: bytes) -> bytes:
    """Parse BinaryCIF bytes and emit assembly wire bytes. Transport helper; prefer
    `Assembly.from_bcif`."""
    ...

def assembly_bytes_to_pdb(bytes: bytes) -> str:
    """Decode assembly wire bytes and emit a PDB string.
    Transport / serialization helper."""
    ...

def deserialize_assembly_bytes(bytes: bytes) -> bytes:
    """Round-trip assembly wire bytes through `Assembly` and back (validation).
    Transport / serialization helper."""
    ...

def assembly_bytes_to_arrays(bytes: bytes) -> AtomArrays:
    """Decode wire bytes to per-atom numpy columns (Biotite-free). Transport
    helper; prefer `Assembly.to_arrays` / `Entity.to_arrays`."""
    ...

# ---------------------------------------------------------------------------
# AtomWorks / Biotite interop free functions
#
# Convert between molex's wire bytes / entities and Biotite `AtomArray` /
# `AtomArrayPlus` objects, and parse structure files through the AtomWorks
# pipeline. `AtomArray` arguments and returns are typed `Any` (Biotite is not
# a typing dependency).
# ---------------------------------------------------------------------------

def entities_to_atom_array(assembly_bytes: bytes) -> Any:
    """Convert assembly wire bytes to a Biotite `AtomArray`."""
    ...

def entities_to_atom_array_plus(assembly_bytes: bytes) -> Any:
    """Convert assembly wire bytes to an AtomWorks `AtomArrayPlus` (signals the
    structure is already fully constructed; skips CCD rebuilding)."""
    ...

def atom_array_to_entities(atom_array: Any) -> bytes:
    """Convert a Biotite `AtomArray` (or `AtomArrayPlus`) back to assembly wire
    bytes."""
    ...

def assembly_bytes_to_atom_array(bytes: bytes) -> Any:
    """Convert assembly wire bytes to a Biotite `AtomArray`."""
    ...

def assembly_bytes_to_atom_array_plus(bytes: bytes) -> Any:
    """Convert assembly wire bytes to an AtomWorks `AtomArrayPlus`."""
    ...

def entities_to_atom_array_parsed(
    assembly_bytes: bytes, source_path: str | None = ...
) -> Any:
    """Convert entities to an `AtomArray`, then run AtomWorks' full cleaning
    pipeline (highest fidelity; slower than `entities_to_atom_array`)."""
    ...

def parse_file_to_entities(file_path: str) -> bytes:
    """Load a structure file through AtomWorks' full parsing pipeline and
    return assembly wire bytes of the cleaned, annotated entities."""
    ...

def parse_file_full(file_path: str) -> Any:
    """Load a structure file through AtomWorks and return a dict with
    `"assembly_bytes"`, `"chain_info"`, and `"assemblies"`."""
    ...
