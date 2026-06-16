//! CIF/STAR text parser.
//!
//! Parses any valid CIF or STAR file into an untyped [`Document`] tree.
//! Handles all value forms: unquoted, single/double-quoted, and semicolon
//! text fields. Endianness and case conventions are preserved in tag names.

use super::dom::{Block, Document, Loop, Value};

/// Errors that can occur during CIF/STAR parsing.
#[derive(Debug, thiserror::Error)]
pub enum CifParseError {
    /// A single- or double-quoted string was never closed.
    #[error("unterminated quoted string at byte offset {0}")]
    UnterminatedQuote(usize),
    /// A semicolon-delimited text field was never closed.
    #[error("unterminated semicolon text field at byte offset {0}")]
    UnterminatedTextField(usize),
}

/// Parse a CIF/STAR text string into a [`Document`].
///
/// # Errors
///
/// Returns [`CifParseError`] if the input contains unterminated quoted strings
/// or semicolon text fields.
pub fn parse(input: &str) -> Result<Document, CifParseError> {
    Parser::new(input).parse_document()
}

// Internal tokenizer / parser

#[derive(Debug)]
enum Token {
    DataBlock(String),
    LoopStart,
    SaveStart(String),
    SaveEnd,
    Tag(String),
    Val(Value),
    Eof,
}

struct Parser<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
    at_line_start: bool,
    pending: Option<Token>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
            at_line_start: true,
            pending: None,
        }
    }

    fn next(&mut self) -> Result<Token, CifParseError> {
        if let Some(t) = self.pending.take() {
            return Ok(t);
        }
        self.scan_token()
    }

    fn push_back(&mut self, token: Token) {
        debug_assert!(self.pending.is_none());
        self.pending = Some(token);
    }

    // Tokenizer

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\r' => self.pos += 1,
                b'\n' => {
                    self.pos += 1;
                    self.at_line_start = true;
                }
                b'#' => {
                    while self.pos < self.bytes.len()
                        && self.bytes[self.pos] != b'\n'
                    {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn scan_token(&mut self) -> Result<Token, CifParseError> {
        self.skip_whitespace_and_comments();
        if self.pos >= self.bytes.len() {
            return Ok(Token::Eof);
        }

        let b = self.bytes[self.pos];

        // Semicolon text field (only valid at line start)
        if b == b';' && self.at_line_start {
            return self.scan_semicolon_text();
        }

        self.at_line_start = false;

        // Quoted strings
        if b == b'\'' || b == b'"' {
            return self.scan_quoted(b);
        }

        // Unquoted token
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            if c.is_ascii_whitespace() || c == b'#' {
                break;
            }
            self.pos += 1;
        }
        Ok(classify_unquoted(&self.input[start..self.pos]))
    }

    fn scan_quoted(&mut self, quote: u8) -> Result<Token, CifParseError> {
        let start = self.pos;
        self.pos += 1; // skip opening quote
        loop {
            if self.pos >= self.bytes.len() {
                return Err(CifParseError::UnterminatedQuote(start));
            }
            if self.bytes[self.pos] == quote {
                // CIF rule: closing quote must be followed by
                // whitespace/comment/EOF
                if self.pos + 1 >= self.bytes.len()
                    || self.bytes[self.pos + 1].is_ascii_whitespace()
                    || self.bytes[self.pos + 1] == b'#'
                {
                    let val = self.input[start + 1..self.pos].to_owned();
                    self.pos += 1; // skip closing quote
                    return Ok(Token::Val(Value::Str(val)));
                }
            }
            self.pos += 1;
        }
    }

    fn scan_semicolon_text(&mut self) -> Result<Token, CifParseError> {
        let start = self.pos;
        self.pos += 1; // skip opening ;
        self.at_line_start = false;
        let content_start = self.pos;

        loop {
            // Advance to next newline
            while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                self.pos += 1;
            }
            if self.pos >= self.bytes.len() {
                return Err(CifParseError::UnterminatedTextField(start));
            }
            self.pos += 1; // skip \n

            // Check if next line starts with ;
            if self.pos < self.bytes.len() && self.bytes[self.pos] == b';' {
                // Content excludes the trailing \n before closing ;
                let content_end = self.pos - 1;
                let text = self.input[content_start..content_end].to_owned();
                self.pos += 1; // skip closing ;
                self.at_line_start = false;
                return Ok(Token::Val(Value::Str(text)));
            }
        }
    }

    // Structure parsing

    fn parse_document(&mut self) -> Result<Document, CifParseError> {
        let mut blocks = Vec::new();
        loop {
            let token = self.next()?;
            match token {
                Token::Eof => break,
                Token::DataBlock(name) => blocks.push(self.parse_block(name)?),
                _ => {} // skip tokens before first data block
            }
        }
        Ok(Document { blocks })
    }

    fn parse_block(&mut self, name: String) -> Result<Block, CifParseError> {
        let mut pairs = Vec::new();
        let mut loops = Vec::new();
        let mut frames = Vec::new();

        loop {
            let token = self.next()?;
            match token {
                Token::Eof | Token::DataBlock(_) => {
                    self.push_back(token);
                    break;
                }
                Token::LoopStart => loops.push(self.parse_loop()?),
                Token::SaveStart(frame_name) => {
                    frames.push(self.parse_block(frame_name)?);
                }
                Token::SaveEnd => break,
                Token::Tag(tag) => {
                    let val_token = self.next()?;
                    match val_token {
                        Token::Val(v) => pairs.push((tag, v)),
                        other => self.push_back(other), // tag without value
                    }
                }
                Token::Val(_) => {} // stray value, skip
            }
        }

        Ok(Block {
            name,
            pairs,
            loops,
            frames,
        })
    }

    fn parse_loop(&mut self) -> Result<Loop, CifParseError> {
        let mut tags = Vec::new();
        let mut values = Vec::new();

        // Collect tags
        loop {
            let token = self.next()?;
            match token {
                Token::Tag(t) => tags.push(t),
                other => {
                    self.push_back(other);
                    break;
                }
            }
        }

        // Collect values
        loop {
            let token = self.next()?;
            match token {
                Token::Val(v) => values.push(v),
                other => {
                    self.push_back(other);
                    break;
                }
            }
        }

        Ok(Loop { tags, values })
    }
}

fn classify_unquoted(s: &str) -> Token {
    if s.len() >= 5 && s[..5].eq_ignore_ascii_case("data_") {
        Token::DataBlock(s[5..].to_owned())
    } else if s.len() == 5 && s.eq_ignore_ascii_case("loop_") {
        Token::LoopStart
    } else if s.len() >= 5 && s[..5].eq_ignore_ascii_case("save_") {
        if s.len() == 5 {
            Token::SaveEnd
        } else {
            Token::SaveStart(s[5..].to_owned())
        }
    } else if s.starts_with('_') {
        Token::Tag(s.to_owned())
    } else if s == "." {
        Token::Val(Value::Inapplicable)
    } else if s == "?" {
        Token::Val(Value::Unknown)
    } else {
        Token::Val(Value::Str(s.to_owned()))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn key_value_pairs() {
        let input = "data_test\n_cell.length_a 50.0\n_cell.length_b 60.0\n";
        let doc = parse(input).unwrap();
        assert_eq!(doc.blocks.len(), 1);
        assert_eq!(doc.blocks[0].name, "test");
        assert_eq!(
            doc.blocks[0].get("_cell.length_a"),
            Some(&Value::Str("50.0".into()))
        );
        assert_eq!(
            doc.blocks[0].get("_cell.length_b"),
            Some(&Value::Str("60.0".into()))
        );
    }

    #[test]
    fn loop_basic() {
        let input = "data_test\nloop_\n_col1\n_col2\na b\nc d\n";
        let doc = parse(input).unwrap();
        let lp = &doc.blocks[0].loops[0];
        assert_eq!(lp.tags, vec!["_col1", "_col2"]);
        assert_eq!(lp.nrows(), 2);
        assert_eq!(lp.values[0], Value::Str("a".into()));
        assert_eq!(lp.values[1], Value::Str("b".into()));
        assert_eq!(lp.values[2], Value::Str("c".into()));
        assert_eq!(lp.values[3], Value::Str("d".into()));
    }

    #[test]
    fn quoted_strings() {
        let input = "data_test\n_tag1 'hello world'\n_tag2 \"another value\"\n";
        let doc = parse(input).unwrap();
        let block = &doc.blocks[0];
        assert_eq!(block.get("_tag1"), Some(&Value::Str("hello world".into())));
        assert_eq!(
            block.get("_tag2"),
            Some(&Value::Str("another value".into()))
        );
    }

    #[test]
    fn semicolon_text_field() {
        let input = "data_test\n_tag1\n;line one\nline two\n;\n";
        let doc = parse(input).unwrap();
        assert_eq!(
            doc.blocks[0].get("_tag1"),
            Some(&Value::Str("line one\nline two".into()))
        );
    }

    #[test]
    fn special_values() {
        let input = "data_test\n_tag1 .\n_tag2 ?\n";
        let doc = parse(input).unwrap();
        assert_eq!(doc.blocks[0].get("_tag1"), Some(&Value::Inapplicable));
        assert_eq!(doc.blocks[0].get("_tag2"), Some(&Value::Unknown));
    }

    #[test]
    fn comments() {
        let input =
            "# file comment\ndata_test\n_tag1 value1 # inline\n_tag2 value2\n";
        let doc = parse(input).unwrap();
        assert_eq!(
            doc.blocks[0].get("_tag1"),
            Some(&Value::Str("value1".into()))
        );
        assert_eq!(
            doc.blocks[0].get("_tag2"),
            Some(&Value::Str("value2".into()))
        );
    }

    #[test]
    fn multiple_blocks() {
        let input = "data_first\n_a 1\ndata_second\n_b 2\n";
        let doc = parse(input).unwrap();
        assert_eq!(doc.blocks.len(), 2);
        assert_eq!(doc.blocks[0].name, "first");
        assert_eq!(doc.blocks[1].name, "second");
        assert_eq!(doc.blocks[0].get("_a"), Some(&Value::Str("1".into())));
        assert_eq!(doc.blocks[1].get("_b"), Some(&Value::Str("2".into())));
    }

    #[test]
    fn columns_reordered() {
        let input = "data_test\nloop_\n_x\n_y\n_z\n1 2 3\n4 5 6\n";
        let doc = parse(input).unwrap();
        let cols = doc.blocks[0].columns(&["_z", "_x"]).unwrap();
        assert_eq!(cols.nrows(), 2);
        assert_eq!(cols.get(0, 0), &Value::Str("3".into()));
        assert_eq!(cols.get(0, 1), &Value::Str("1".into()));
        assert_eq!(cols.get(1, 0), &Value::Str("6".into()));
        assert_eq!(cols.get(1, 1), &Value::Str("4".into()));
    }

    #[test]
    fn columns_missing_tag() {
        let input = "data_test\nloop_\n_x\n_y\n1 2\n";
        let doc = parse(input).unwrap();
        assert!(doc.blocks[0].columns(&["_x", "_z"]).is_none());
    }

    #[test]
    fn case_insensitive_lookup() {
        let input = "data_test\n_Cell.Length_A 50.0\nloop_\n_Atom.X\n10\n";
        let doc = parse(input).unwrap();
        assert!(doc.blocks[0].get("_cell.length_a").is_some());
        assert!(doc.blocks[0].find_loop("_atom.x").is_some());
    }

    #[test]
    fn column_iterator() {
        let input = "data_test\nloop_\n_a\n_b\n1 2\n3 4\n";
        let doc = parse(input).unwrap();
        let vals: Vec<_> = doc.blocks[0].column("_b").unwrap().collect();
        assert_eq!(
            vals,
            vec![&Value::Str("2".into()), &Value::Str("4".into())]
        );
    }

    #[test]
    fn row_iterator() {
        let input = "data_test\nloop_\n_a\n_b\n_c\n1 2 3\n4 5 6\n";
        let doc = parse(input).unwrap();
        let cols = doc.blocks[0].columns(&["_c", "_a"]).unwrap();
        let rows: Vec<_> = cols.iter().collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            vec![&Value::Str("3".into()), &Value::Str("1".into())]
        );
        assert_eq!(
            rows[1],
            vec![&Value::Str("6".into()), &Value::Str("4".into())]
        );
    }

    #[test]
    fn value_parsing() {
        assert_eq!(Value::Str("42".into()).as_i32(), Some(42));
        assert_eq!(Value::Str("2.72".into()).as_f64(), Some(2.72));
        assert_eq!(Value::Str("50.123(4)".into()).as_f64(), Some(50.123));
        assert_eq!(Value::Inapplicable.as_f64(), None);
        assert_eq!(Value::Unknown.as_str(), None);
    }

    #[test]
    fn empty_document() {
        let doc = parse("").unwrap();
        assert!(doc.blocks.is_empty());
    }

    #[test]
    fn save_frames() {
        let input = "data_dict\nsave_myframe\n_tag val\nsave_\n_other after\n";
        let doc = parse(input).unwrap();
        assert_eq!(doc.blocks[0].frames.len(), 1);
        assert_eq!(doc.blocks[0].frames[0].name, "myframe");
        assert_eq!(
            doc.blocks[0].frames[0].get("_tag"),
            Some(&Value::Str("val".into()))
        );
        assert_eq!(
            doc.blocks[0].get("_other"),
            Some(&Value::Str("after".into()))
        );
    }

    #[test]
    fn unterminated_quote() {
        let err = parse("data_test\n_tag 'unterminated").unwrap_err();
        assert!(matches!(err, CifParseError::UnterminatedQuote(_)));
    }

    #[test]
    fn loop_with_special_values() {
        let input = "data_test\nloop_\n_a\n_b\nfoo .\nbar ?\n";
        let doc = parse(input).unwrap();
        let lp = &doc.blocks[0].loops[0];
        assert_eq!(lp.nrows(), 2);
        assert_eq!(lp.values[1], Value::Inapplicable);
        assert_eq!(lp.values[3], Value::Unknown);
    }

    #[test]
    fn quote_with_embedded_quote() {
        // Single quote inside double quotes is fine
        let input = "data_test\n_tag \"it's fine\"\n";
        let doc = parse(input).unwrap();
        assert_eq!(
            doc.blocks[0].get("_tag"),
            Some(&Value::Str("it's fine".into()))
        );
    }

    #[test]
    fn realistic_mmcif_snippet() {
        let input = r"data_1ABC
_cell.length_a   50.000
_cell.length_b   60.000
_cell.length_c   70.000
_cell.angle_alpha 90.00
_cell.angle_beta  90.00
_cell.angle_gamma 90.00
_symmetry.space_group_name_H-M 'P 21 21 21'
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM N   ALA A 1 10.000 20.000 30.000 N 1.00 15.0
ATOM CA  ALA A 1 11.000 21.000 31.000 C 1.00 16.0
ATOM C   ALA A 1 12.000 22.000 32.000 C 1.00 17.0
";
        let doc = parse(input).unwrap();
        let block = &doc.blocks[0];
        assert_eq!(block.name, "1ABC");

        // Cell parameters
        assert_eq!(block.get("_cell.length_a").unwrap().as_f64(), Some(50.0));

        // Space group
        assert_eq!(
            block
                .get("_symmetry.space_group_name_H-M")
                .unwrap()
                .as_str(),
            Some("P 21 21 21")
        );

        // Atom loop
        let cols = block
            .columns(&[
                "_atom_site.label_atom_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
            ])
            .unwrap();
        assert_eq!(cols.nrows(), 3);
        assert_eq!(cols.get(1, 0).as_str(), Some("CA"));
        assert_eq!(cols.get(1, 1).as_f64(), Some(11.0));
    }
}
