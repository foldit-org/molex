//! Opaque entity identifier with controlled allocation.
//!
//! [`EntityId`] can only be created through an [`EntityIdAllocator`],
//! preventing duplicate IDs across the entity system.

/// Opaque entity identifier.
///
/// Cannot be constructed directly, only through [`EntityIdAllocator`].
/// Use [`raw()`](EntityId::raw) for serialization or GPU buffer usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct EntityId(u32);

impl EntityId {
    /// Raw numeric value for serialization, GPU buffers, or display.
    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

    /// Rebuild a host-assigned id from its raw value, for wire decode.
    ///
    /// Unlike [`EntityIdAllocator::from_raw`], this touches no allocator
    /// and advances no counter: the wire only ever carries ids the host
    /// already minted, so no uniqueness bookkeeping is needed here.
    #[must_use]
    pub fn from_raw(raw: u32) -> EntityId {
        EntityId(raw)
    }
}

impl std::ops::Deref for EntityId {
    type Target = u32;

    fn deref(&self) -> &u32 {
        &self.0
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Monotonic entity ID allocator. The only way to create an [`EntityId`].
///
/// Guarantees uniqueness within the allocator's lifetime. Each call to
/// [`allocate`](Self::allocate) produces a new, never-before-seen ID.
#[derive(Debug)]
pub struct EntityIdAllocator {
    next: u32,
}

impl EntityIdAllocator {
    /// Create a new allocator starting from ID 0.
    #[must_use]
    pub fn new() -> Self {
        Self { next: 0 }
    }

    /// Allocate the next unique entity ID.
    pub fn allocate(&mut self) -> EntityId {
        let id = EntityId(self.next);
        self.next += 1;
        id
    }

    /// Reconstruct an [`EntityId`] from a raw value (deserialization).
    ///
    /// Advances the internal counter past this value to prevent future
    /// collisions.
    pub fn from_raw(&mut self, raw: u32) -> EntityId {
        self.next = self.next.max(raw + 1);
        EntityId(raw)
    }
}

impl Default for EntityIdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_allocation() {
        let mut alloc = EntityIdAllocator::new();
        assert_eq!(alloc.allocate().raw(), 0);
        assert_eq!(alloc.allocate().raw(), 1);
        assert_eq!(alloc.allocate().raw(), 2);
    }

    #[test]
    fn from_raw_advances_counter() {
        let mut alloc = EntityIdAllocator::new();
        let id = alloc.from_raw(10);
        assert_eq!(id.raw(), 10);
        // Next allocation should be past 10
        assert_eq!(alloc.allocate().raw(), 11);
    }

    #[test]
    fn from_raw_no_regression() {
        let mut alloc = EntityIdAllocator::new();
        let _ = alloc.allocate(); // 0
        let _ = alloc.allocate(); // 1
        let _ = alloc.allocate(); // 2
                                  // from_raw with a lower value shouldn't regress the counter
        let id = alloc.from_raw(1);
        assert_eq!(id.raw(), 1);
        assert_eq!(alloc.allocate().raw(), 3);
    }

    #[test]
    fn equality_and_hash() {
        let mut alloc = EntityIdAllocator::new();
        let a = alloc.allocate();
        let b = alloc.allocate();
        let a2 = alloc.from_raw(0);
        assert_ne!(a, b);
        assert_eq!(a, a2);
    }
}
