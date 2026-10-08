use super::State;
use candle::Result;
use std::io::{self, Write};

pub(super) struct BoundedText {
    bytes: Vec<u8>,
    limit: usize,
}

impl BoundedText {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }
    pub(super) fn push(&mut self, text: &str) -> Result<()> {
        self.write_all(text.as_bytes()).map_err(candle::Error::wrap)
    }
    pub(super) fn finish(self) -> Result<String> {
        String::from_utf8(self.bytes).map_err(candle::Error::wrap)
    }
    fn indent(&mut self, depth: usize) -> Result<()> {
        for _ in 0..depth {
            self.push("  ")?;
        }
        Ok(())
    }
    fn string(&mut self, text: &str) -> Result<()> {
        serde_json::to_writer(self, text).map_err(candle::Error::wrap)
    }
}

impl Write for BoundedText {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|size| size > self.limit)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "d1 rendered prompt exceeds its byte limit",
            ));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn render(state: &State, text: &mut BoundedText, depth: usize) -> Result<()> {
    if depth > 64 {
        candle::bail!("d1 state exceeds depth 64")
    }
    match state {
        State::Null => text.push("null"),
        State::Bool(value) => text.push(if *value { "true" } else { "false" }),
        State::Number(value) => text.push(&python_number(value)?),
        State::Text(value) => text.string(value),
        State::Array(values) => {
            text.push("[")?;
            for (index, value) in values.iter().enumerate() {
                text.push(if index == 0 { "\n" } else { ",\n" })?;
                text.indent(depth + 1)?;
                render(value, text, depth + 1)?;
            }
            if !values.is_empty() {
                text.push("\n")?;
                text.indent(depth)?;
            }
            text.push("]")
        }
        State::Object(values) => {
            text.push("{")?;
            let mut names = std::collections::HashSet::new();
            for (index, (name, value)) in values.0.iter().enumerate() {
                if !names.insert(name) {
                    candle::bail!("d1 state contains a duplicate key")
                }
                text.push(if index == 0 { "\n" } else { ",\n" })?;
                text.indent(depth + 1)?;
                text.string(name)?;
                text.push(": ")?;
                render(value, text, depth + 1)?;
            }
            if !values.0.is_empty() {
                text.push("\n")?;
                text.indent(depth)?;
            }
            text.push("}")
        }
    }
}

fn python_number(number: &serde_json::Number) -> Result<String> {
    let raw = number.to_string();
    if !number.is_f64() {
        return Ok(raw);
    }
    let (sign, unsigned) = if let Some(rest) = raw.strip_prefix('-') {
        ("-", rest)
    } else {
        ("", raw.as_str())
    };
    let (mantissa, exponent) = match unsigned
        .split_once('e')
        .or_else(|| unsigned.split_once('E'))
    {
        Some((mantissa, exponent)) => (
            mantissa,
            exponent.parse::<i32>().map_err(candle::Error::wrap)?,
        ),
        None => (unsigned, 0),
    };
    let point = mantissa.find('.').unwrap_or(mantissa.len());
    let digits = mantissa.replace('.', "");
    let Some(first) = digits.find(|c: char| c != '0') else {
        return Ok(format!("{sign}0.0"));
    };
    let exponent = exponent + point as i32 - first as i32 - 1;
    let digits = digits[first..].trim_end_matches('0');
    if !(-4..16).contains(&exponent) {
        let (head, tail) = digits.split_at(1);
        return Ok(format!(
            "{sign}{head}{}{tail}e{exponent:+03}",
            if tail.is_empty() { "" } else { "." }
        ));
    }
    let point = exponent + 1;
    if point <= 0 {
        Ok(format!("{sign}0.{}{digits}", "0".repeat((-point) as usize)))
    } else if point as usize >= digits.len() {
        Ok(format!(
            "{sign}{digits}{}.0",
            "0".repeat(point as usize - digits.len())
        ))
    } else {
        let (head, tail) = digits.split_at(point as usize);
        Ok(format!("{sign}{head}.{tail}"))
    }
}
