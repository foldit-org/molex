//! BinaryCIF encoding chain decoder and supporting types.
//!
//! This module contains the `decode_*` functions that implement the various
//! BinaryCIF encoding schemes (ByteArray, FixedPoint, RunLength, Delta,
//! IntegerPacking, IntervalQuantization, StringArray) as well as the
//! lightweight `MsgVal` MessagePack value tree and `ColData` column types.

use std::io::Read;

use crate::ops::codec::AdapterError;

// Lightweight MessagePack value tree

#[derive(Debug, Clone)]
pub(crate) enum MsgVal {
    Nil,
    Bool(bool),
    Int(i64),
    Uint(u64),
    F32(f32),
    F64(f64),
    Str(String),
    Bin(Vec<u8>),
    Array(Vec<MsgVal>),
    Map(Vec<(MsgVal, MsgVal)>),
}

impl MsgVal {
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            MsgVal::Str(s) => Some(s),
            _ => None,
        }
    }

    #[allow(
        clippy::cast_possible_wrap,
        reason = "u64->i64 wrap is acceptable for msgpack values"
    )]
    pub(crate) fn as_i64(&self) -> Option<i64> {
        match self {
            MsgVal::Int(v) => Some(*v),
            MsgVal::Uint(v) => Some(*v as i64),
            _ => None,
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "precision loss is acceptable for i64/u64->f64 in molecular \
                  data"
    )]
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            MsgVal::F64(v) => Some(*v),
            MsgVal::F32(v) => Some(f64::from(*v)),
            MsgVal::Int(v) => Some(*v as f64),
            MsgVal::Uint(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self {
            MsgVal::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[MsgVal]> {
        match self {
            MsgVal::Array(a) => Some(a),
            _ => None,
        }
    }

    pub(crate) fn as_bin(&self) -> Option<&[u8]> {
        match self {
            MsgVal::Bin(b) => Some(b),
            _ => None,
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<&MsgVal> {
        let MsgVal::Map(pairs) = self else {
            return None;
        };
        for (k, v) in pairs {
            if let MsgVal::Str(s) = k {
                if s == key {
                    return Some(v);
                }
            }
        }
        None
    }
}

// MessagePack decoder

pub(crate) fn decode_msgpack(data: &[u8]) -> Result<MsgVal, AdapterError> {
    let mut cursor = std::io::Cursor::new(data);
    read_value(&mut cursor)
}

fn read_bytes<const N: usize>(
    rd: &mut std::io::Cursor<&[u8]>,
) -> Result<[u8; N], AdapterError> {
    let mut buf = [0u8; N];
    rd.read_exact(&mut buf).map_err(|e| {
        AdapterError::InvalidFormat(format!("msgpack read {N} bytes: {e}"))
    })?;
    Ok(buf)
}

#[allow(
    clippy::too_many_lines,
    reason = "msgpack format requires exhaustive marker matching"
)]
fn read_value(rd: &mut std::io::Cursor<&[u8]>) -> Result<MsgVal, AdapterError> {
    use rmp::Marker;

    let marker = rmp::decode::read_marker(rd).map_err(|e| {
        AdapterError::InvalidFormat(format!("msgpack marker: {e:?}"))
    })?;

    match marker {
        Marker::Null => Ok(MsgVal::Nil),
        Marker::True => Ok(MsgVal::Bool(true)),
        Marker::False => Ok(MsgVal::Bool(false)),

        Marker::FixPos(v) => Ok(MsgVal::Uint(u64::from(v))),
        Marker::FixNeg(v) => Ok(MsgVal::Int(i64::from(v))),

        Marker::U8 => Ok(MsgVal::Uint(u64::from(read_bytes::<1>(rd)?[0]))),
        Marker::U16 => {
            Ok(MsgVal::Uint(u64::from(u16::from_be_bytes(read_bytes(rd)?))))
        }
        Marker::U32 => {
            Ok(MsgVal::Uint(u64::from(u32::from_be_bytes(read_bytes(rd)?))))
        }
        Marker::U64 => Ok(MsgVal::Uint(u64::from_be_bytes(read_bytes(rd)?))),
        Marker::I8 => {
            Ok(MsgVal::Int(i64::from(i8::from_be_bytes(read_bytes(rd)?))))
        }
        Marker::I16 => {
            Ok(MsgVal::Int(i64::from(i16::from_be_bytes(read_bytes(rd)?))))
        }
        Marker::I32 => {
            Ok(MsgVal::Int(i64::from(i32::from_be_bytes(read_bytes(rd)?))))
        }
        Marker::I64 => Ok(MsgVal::Int(i64::from_be_bytes(read_bytes(rd)?))),
        Marker::F32 => Ok(MsgVal::F32(f32::from_be_bytes(read_bytes(rd)?))),
        Marker::F64 => Ok(MsgVal::F64(f64::from_be_bytes(read_bytes(rd)?))),

        Marker::FixStr(len) => read_string(rd, usize::from(len)),
        Marker::Str8 => {
            let len = usize::from(read_bytes::<1>(rd)?[0]);
            read_string(rd, len)
        }
        Marker::Str16 => {
            let len = usize::from(u16::from_be_bytes(read_bytes(rd)?));
            read_string(rd, len)
        }
        Marker::Str32 => {
            let len = u32::from_be_bytes(read_bytes(rd)?) as usize;
            read_string(rd, len)
        }

        Marker::Bin8 => {
            let len = usize::from(read_bytes::<1>(rd)?[0]);
            read_bin(rd, len)
        }
        Marker::Bin16 => {
            let len = usize::from(u16::from_be_bytes(read_bytes(rd)?));
            read_bin(rd, len)
        }
        Marker::Bin32 => {
            let len = u32::from_be_bytes(read_bytes(rd)?) as usize;
            read_bin(rd, len)
        }

        Marker::FixArray(len) => read_array(rd, usize::from(len)),
        Marker::Array16 => {
            let len = usize::from(u16::from_be_bytes(read_bytes(rd)?));
            read_array(rd, len)
        }
        Marker::Array32 => {
            let len = u32::from_be_bytes(read_bytes(rd)?) as usize;
            read_array(rd, len)
        }

        Marker::FixMap(len) => read_map(rd, usize::from(len)),
        Marker::Map16 => {
            let len = usize::from(u16::from_be_bytes(read_bytes(rd)?));
            read_map(rd, len)
        }
        Marker::Map32 => {
            let len = u32::from_be_bytes(read_bytes(rd)?) as usize;
            read_map(rd, len)
        }

        other => Err(AdapterError::InvalidFormat(format!(
            "Unsupported msgpack marker: {other:?}"
        ))),
    }
}

fn read_string(
    rd: &mut std::io::Cursor<&[u8]>,
    len: usize,
) -> Result<MsgVal, AdapterError> {
    let mut buf = vec![0u8; len];
    rd.read_exact(&mut buf).map_err(|e| {
        AdapterError::InvalidFormat(format!("msgpack string read: {e}"))
    })?;
    let s = String::from_utf8(buf).map_err(|e| {
        AdapterError::InvalidFormat(format!("msgpack string utf8: {e}"))
    })?;
    Ok(MsgVal::Str(s))
}

fn read_bin(
    rd: &mut std::io::Cursor<&[u8]>,
    len: usize,
) -> Result<MsgVal, AdapterError> {
    let mut buf = vec![0u8; len];
    rd.read_exact(&mut buf).map_err(|e| {
        AdapterError::InvalidFormat(format!("msgpack bin read: {e}"))
    })?;
    Ok(MsgVal::Bin(buf))
}

fn read_array(
    rd: &mut std::io::Cursor<&[u8]>,
    len: usize,
) -> Result<MsgVal, AdapterError> {
    let mut arr = Vec::with_capacity(len);
    for _ in 0..len {
        arr.push(read_value(rd)?);
    }
    Ok(MsgVal::Array(arr))
}

fn read_map(
    rd: &mut std::io::Cursor<&[u8]>,
    len: usize,
) -> Result<MsgVal, AdapterError> {
    let mut pairs = Vec::with_capacity(len);
    for _ in 0..len {
        let k = read_value(rd)?;
        let v = read_value(rd)?;
        pairs.push((k, v));
    }
    Ok(MsgVal::Map(pairs))
}

// BinaryCIF encoding chain decoder

#[derive(Debug)]
pub(crate) enum ColData {
    IntArray(Vec<i32>),
    FloatArray(Vec<f64>),
    StringArray(Vec<String>),
    Bytes(Vec<u8>),
}

pub(crate) fn decode_column(
    data_node: &MsgVal,
) -> Result<ColData, AdapterError> {
    let raw_bytes =
        data_node
            .get("data")
            .and_then(MsgVal::as_bin)
            .ok_or_else(|| {
                AdapterError::InvalidFormat(
                    "Column missing 'data' bytes".into(),
                )
            })?;

    let encodings = data_node
        .get("encoding")
        .and_then(MsgVal::as_array)
        .ok_or_else(|| {
            AdapterError::InvalidFormat(
                "Column missing 'encoding' array".into(),
            )
        })?;

    if encodings.is_empty() {
        return Ok(ColData::Bytes(raw_bytes.to_vec()));
    }

    let first_kind = encodings[0]
        .get("kind")
        .and_then(MsgVal::as_str)
        .unwrap_or("");

    if first_kind == "StringArray" {
        return decode_string_array_column(
            raw_bytes,
            &encodings[0],
            &encodings[1..],
        );
    }

    let mut current = ColData::Bytes(raw_bytes.to_vec());
    for enc in encodings.iter().rev() {
        let kind =
            enc.get("kind").and_then(MsgVal::as_str).ok_or_else(|| {
                AdapterError::InvalidFormat("Encoding missing 'kind'".into())
            })?;

        current = match kind {
            "ByteArray" => decode_byte_array(current, enc)?,
            "FixedPoint" => decode_fixed_point(current, enc)?,
            "IntervalQuantization" => {
                decode_interval_quantization(current, enc)?
            }
            "RunLength" => decode_run_length(current, enc)?,
            "Delta" => decode_delta(current, enc)?,
            "IntegerPacking" => decode_integer_packing(current, enc)?,
            other => {
                return Err(AdapterError::InvalidFormat(format!(
                    "Unknown encoding kind: {other}"
                )))
            }
        };
    }

    Ok(current)
}

#[allow(
    clippy::too_many_lines,
    reason = "binary format type dispatch requires exhaustive matching"
)]
fn decode_byte_array(
    input: ColData,
    enc: &MsgVal,
) -> Result<ColData, AdapterError> {
    let ColData::Bytes(bytes) = input else {
        return Err(AdapterError::InvalidFormat(
            "ByteArray expects bytes input".into(),
        ));
    };

    #[allow(
        clippy::cast_possible_truncation,
        reason = "type_id is a BinaryCIF type tag (0..33)"
    )]
    #[allow(
        clippy::cast_sign_loss,
        reason = "type_id is a non-negative BinaryCIF type tag"
    )]
    let type_id = enc.get("type").and_then(MsgVal::as_i64).ok_or_else(|| {
        AdapterError::InvalidFormat("ByteArray missing 'type'".into())
    })? as u8;

    match type_id {
        1 => Ok(ColData::IntArray(
            bytes.iter().map(|&b| i32::from(b.cast_signed())).collect(),
        )),
        2 => Ok(ColData::IntArray(
            bytes
                .chunks_exact(2)
                .map(|c| i32::from(i16::from_le_bytes([c[0], c[1]])))
                .collect(),
        )),
        3 => Ok(ColData::IntArray(
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )),
        4 => Ok(ColData::IntArray(
            bytes.iter().map(|&b| i32::from(b)).collect(),
        )),
        5 => Ok(ColData::IntArray(
            bytes
                .chunks_exact(2)
                .map(|c| i32::from(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        )),
        #[allow(
            clippy::cast_possible_wrap,
            reason = "u32->i32 wrap matches BinaryCIF spec for type 6"
        )]
        6 => Ok(ColData::IntArray(
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i32)
                .collect(),
        )),
        32 => Ok(ColData::FloatArray(
            bytes
                .chunks_exact(4)
                .map(|c| {
                    f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                })
                .collect(),
        )),
        33 => Ok(ColData::FloatArray(
            bytes
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([
                        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                    ])
                })
                .collect(),
        )),
        _ => Err(AdapterError::InvalidFormat(format!(
            "Unknown ByteArray type: {type_id}"
        ))),
    }
}

fn decode_fixed_point(
    input: ColData,
    enc: &MsgVal,
) -> Result<ColData, AdapterError> {
    let ColData::IntArray(ints) = input else {
        return Err(AdapterError::InvalidFormat(
            "FixedPoint expects int array".into(),
        ));
    };

    let factor =
        enc.get("factor").and_then(MsgVal::as_f64).ok_or_else(|| {
            AdapterError::InvalidFormat("FixedPoint missing 'factor'".into())
        })?;

    let inv = 1.0 / factor;
    Ok(ColData::FloatArray(
        ints.iter().map(|&v| f64::from(v) * inv).collect(),
    ))
}

fn decode_interval_quantization(
    input: ColData,
    enc: &MsgVal,
) -> Result<ColData, AdapterError> {
    let ColData::IntArray(ints) = input else {
        return Err(AdapterError::InvalidFormat(
            "IntervalQuantization expects int array".into(),
        ));
    };

    let min = enc.get("min").and_then(MsgVal::as_f64).ok_or_else(|| {
        AdapterError::InvalidFormat("IntervalQuantization missing 'min'".into())
    })?;
    let max = enc.get("max").and_then(MsgVal::as_f64).ok_or_else(|| {
        AdapterError::InvalidFormat("IntervalQuantization missing 'max'".into())
    })?;
    #[allow(
        clippy::cast_precision_loss,
        reason = "numSteps is a small integer, no meaningful precision loss"
    )]
    let num_steps =
        enc.get("numSteps")
            .and_then(MsgVal::as_i64)
            .ok_or_else(|| {
                AdapterError::InvalidFormat(
                    "IntervalQuantization missing 'numSteps'".into(),
                )
            })? as f64;

    let delta = (max - min) / (num_steps - 1.0);
    Ok(ColData::FloatArray(
        ints.iter()
            .map(|&v| f64::from(v).mul_add(delta, min))
            .collect(),
    ))
}

/// Cap on the cumulative output count of a single `RunLength` decode.
///
/// Encoded `_atom_site` columns at RCSB-published structures top out near
/// 10M rows; the bound here is two orders of magnitude beyond that, low
/// enough to prevent unbounded allocation from a crafted input.
const MAX_RUN_LENGTH_OUTPUT: usize = 1_000_000_000;

fn decode_run_length(
    input: ColData,
    enc: &MsgVal,
) -> Result<ColData, AdapterError> {
    let ColData::IntArray(ints) = input else {
        return Err(AdapterError::InvalidFormat(
            "RunLength expects int array".into(),
        ));
    };

    if ints.len() % 2 != 0 {
        return Err(AdapterError::InvalidFormat(
            "RunLength array length must be even".into(),
        ));
    }

    let expected: Option<usize> = enc
        .get("srcSize")
        .and_then(MsgVal::as_i64)
        .and_then(|n| usize::try_from(n).ok());

    let mut total: usize = 0;
    for pair in ints.chunks_exact(2) {
        if pair[1] < 0 {
            return Err(AdapterError::InvalidFormat(
                "RunLength: negative count".into(),
            ));
        }
        #[allow(clippy::cast_sign_loss, reason = "checked >= 0 above")]
        let count = pair[1] as usize;
        total = total.checked_add(count).ok_or_else(|| {
            AdapterError::InvalidFormat(
                "RunLength: cumulative count overflows usize".into(),
            )
        })?;
        if total > MAX_RUN_LENGTH_OUTPUT {
            return Err(AdapterError::InvalidFormat(format!(
                "RunLength output exceeds {MAX_RUN_LENGTH_OUTPUT} entries"
            )));
        }
    }
    if let Some(expected) = expected {
        if expected > MAX_RUN_LENGTH_OUTPUT {
            return Err(AdapterError::InvalidFormat(format!(
                "RunLength srcSize {expected} exceeds bound"
            )));
        }
        if total != expected {
            return Err(AdapterError::InvalidFormat(format!(
                "RunLength srcSize {expected} disagrees with sum-of-counts \
                 {total}"
            )));
        }
    }

    let mut result = Vec::with_capacity(total);
    for pair in ints.chunks_exact(2) {
        let value = pair[0];
        #[allow(clippy::cast_sign_loss, reason = "non-negative verified above")]
        let count = pair[1] as usize;
        result.extend(std::iter::repeat_n(value, count));
    }
    Ok(ColData::IntArray(result))
}

fn decode_delta(input: ColData, enc: &MsgVal) -> Result<ColData, AdapterError> {
    let ColData::IntArray(mut ints) = input else {
        return Err(AdapterError::InvalidFormat(
            "Delta expects int array".into(),
        ));
    };

    #[allow(
        clippy::cast_possible_truncation,
        reason = "delta origin fits in i32 per BinaryCIF spec"
    )]
    #[allow(
        clippy::cast_possible_wrap,
        reason = "delta origin fits in i32 per BinaryCIF spec"
    )]
    let origin = enc.get("origin").and_then(MsgVal::as_i64).unwrap_or(0) as i32;

    if !ints.is_empty() {
        ints[0] += origin;
        for i in 1..ints.len() {
            ints[i] += ints[i - 1];
        }
    }
    Ok(ColData::IntArray(ints))
}

/// Continuation sentinels for an IntegerPacking width.
///
/// BinaryCIF has two distinct IntegerPacking variants. Unsigned packing
/// saturates only at an upper limit (`0xFF`/`0xFFFF` for byteCount 1/2); a
/// `0` is a legitimate value, never a continuation marker. Signed packing
/// saturates at both an upper (`0x7F`/`0x7FFF`) and a lower
/// (`-upper - 1`, i.e. `-0x80`/`-0x8000`) limit. A token equal to a sentinel
/// means "this width's max magnitude was reached, keep accumulating."
struct PackingLimits {
    upper: i32,
    /// `None` for unsigned widths, which have no lower sentinel.
    lower: Option<i32>,
}

/// Extract integer-packing parameters from a BinaryCIF encoding node.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "byteCount and srcSize are small non-negative per spec"
)]
fn int_packing_params(
    enc: &MsgVal,
) -> Result<(usize, PackingLimits), AdapterError> {
    let byte_count =
        enc.get("byteCount")
            .and_then(MsgVal::as_i64)
            .ok_or_else(|| {
                AdapterError::InvalidFormat(
                    "IntegerPacking missing 'byteCount'".into(),
                )
            })?;
    let src_size =
        enc.get("srcSize").and_then(MsgVal::as_i64).ok_or_else(|| {
            AdapterError::InvalidFormat(
                "IntegerPacking missing 'srcSize'".into(),
            )
        })? as usize;
    let is_unsigned = enc
        .get("isUnsigned")
        .and_then(MsgVal::as_bool)
        .unwrap_or(false);

    let limits = match (is_unsigned, byte_count) {
        (true, 1) => PackingLimits {
            upper: 0xFF,
            lower: None,
        },
        (true, 2) => PackingLimits {
            upper: 0xFFFF,
            lower: None,
        },
        (true, 4) => PackingLimits {
            upper: i32::MAX,
            lower: None,
        },
        (false, 1) => PackingLimits {
            upper: 0x7F,
            lower: Some(-0x80),
        },
        (false, 2) => PackingLimits {
            upper: 0x7FFF,
            lower: Some(-0x8000),
        },
        (false, 4) => PackingLimits {
            upper: i32::MAX,
            lower: Some(i32::MIN),
        },
        _ => {
            return Err(AdapterError::InvalidFormat(format!(
                "IntegerPacking unsupported byteCount={byte_count}"
            )))
        }
    };
    Ok((src_size, limits))
}

fn decode_integer_packing(
    input: ColData,
    enc: &MsgVal,
) -> Result<ColData, AdapterError> {
    let ColData::IntArray(packed) = input else {
        return Err(AdapterError::InvalidFormat(
            "IntegerPacking expects int array".into(),
        ));
    };

    let (src_size, limits) = int_packing_params(enc)?;
    let is_sentinel = |t: i32| t == limits.upper || limits.lower == Some(t);
    let mut result = Vec::with_capacity(src_size);
    let mut i = 0;

    while i < packed.len() && result.len() < src_size {
        let mut value: i32 = 0;
        let mut t = packed[i];
        while is_sentinel(t) {
            value += t;
            i += 1;
            if i >= packed.len() {
                break;
            }
            t = packed[i];
        }
        value += t;
        i += 1;
        result.push(value);
    }

    Ok(ColData::IntArray(result))
}

#[allow(
    clippy::too_many_lines,
    reason = "string array decoding has inherent complexity from the \
              BinaryCIF spec"
)]
fn decode_string_array_column(
    raw_bytes: &[u8],
    sa_enc: &MsgVal,
    remaining_encodings: &[MsgVal],
) -> Result<ColData, AdapterError> {
    let string_data = sa_enc
        .get("stringData")
        .and_then(MsgVal::as_str)
        .ok_or_else(|| {
            AdapterError::InvalidFormat(
                "StringArray missing 'stringData'".into(),
            )
        })?;

    let offset_bytes = sa_enc
        .get("offsets")
        .and_then(MsgVal::as_bin)
        .ok_or_else(|| {
            AdapterError::InvalidFormat("StringArray missing 'offsets'".into())
        })?;
    let offset_encoding = sa_enc
        .get("offsetEncoding")
        .and_then(MsgVal::as_array)
        .ok_or_else(|| {
            AdapterError::InvalidFormat(
                "StringArray missing 'offsetEncoding'".into(),
            )
        })?;

    let offset_data_node = build_encoded_data(offset_bytes, offset_encoding);
    let ColData::IntArray(offsets) = decode_column(&offset_data_node)? else {
        return Err(AdapterError::InvalidFormat(
            "StringArray offsets must decode to int array".into(),
        ));
    };

    let mut strings: Vec<&str> = Vec::with_capacity(if offsets.is_empty() {
        0
    } else {
        offsets.len() - 1
    });
    for w in offsets.windows(2) {
        #[allow(
            clippy::cast_sign_loss,
            reason = "offsets are non-negative indices into string data"
        )]
        let start = w[0] as usize;
        #[allow(
            clippy::cast_sign_loss,
            reason = "offsets are non-negative indices into string data"
        )]
        let end = w[1] as usize;
        if end > string_data.len() || start > end {
            return Err(AdapterError::InvalidFormat(
                "StringArray offset out of bounds".into(),
            ));
        }
        strings.push(&string_data[start..end]);
    }

    let data_encoding = sa_enc
        .get("dataEncoding")
        .and_then(MsgVal::as_array)
        .ok_or_else(|| {
        AdapterError::InvalidFormat("StringArray missing 'dataEncoding'".into())
    })?;

    let mut index_encodings = Vec::new();
    index_encodings.extend_from_slice(data_encoding);
    index_encodings.extend_from_slice(remaining_encodings);

    let index_data_node = build_encoded_data(raw_bytes, &index_encodings);
    let ColData::IntArray(indices) = decode_column(&index_data_node)? else {
        return Err(AdapterError::InvalidFormat(
            "StringArray indices must decode to int array".into(),
        ));
    };

    let result: Vec<String> = indices
        .iter()
        .map(|&idx| {
            #[allow(
                clippy::cast_sign_loss,
                reason = "indices are non-negative by spec"
            )]
            let idx = idx as usize;
            if idx < strings.len() {
                strings[idx].to_owned()
            } else {
                String::new()
            }
        })
        .collect();

    Ok(ColData::StringArray(result))
}

fn build_encoded_data(bytes: &[u8], encodings: &[MsgVal]) -> MsgVal {
    MsgVal::Map(vec![
        (MsgVal::Str("data".into()), MsgVal::Bin(bytes.to_vec())),
        (
            MsgVal::Str("encoding".into()),
            MsgVal::Array(encodings.to_vec()),
        ),
    ])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "test code")]
mod tests {
    use std::path::PathBuf;

    use crate::adapters::bcif::bcif_to_entities;
    use crate::entity::molecule::MoleculeEntity;

    /// Real RCSB `.bcif` exercises unsigned IntegerPacking (`auth_seq_id`,
    /// `label_seq_id`). A leading `0` data byte must decode as the value `0`,
    /// not as a continuation sentinel; getting that wrong shifts the decoded
    /// array by one and trips the downstream RunLength validators, which is
    /// how this file used to fail to decode at all. 1UBQ is 660 atoms
    /// (biotite) including 58 waters.
    #[test]
    fn rcsb_1ubq_decodes_660_atoms() {
        let path: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "tests",
            "data",
            "bcif",
            "1ubq.bcif",
        ]
        .iter()
        .collect();

        let bytes = std::fs::read(&path).expect("read 1ubq.bcif fixture");
        let entities = bcif_to_entities(&bytes).expect("1ubq.bcif must decode");
        let atoms: usize =
            entities.iter().map(MoleculeEntity::atom_count).sum();
        assert_eq!(atoms, 660);
    }
}
