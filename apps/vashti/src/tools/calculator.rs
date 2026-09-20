use fasteval::Evaler;
use serde::Deserialize;
use serde_json::{Value, json};

use super::service::{TOOL_CALCULATE, TOOL_CONVERT_UNITS};

const MAX_EXPRESSION_BYTES: usize = 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Calculation {
    expression: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Conversion {
    value: f64,
    from_unit: String,
    to_unit: String,
}

pub fn execute(name: &str, arguments: &Value) -> Result<String, String> {
    let result = match name {
        TOOL_CALCULATE => {
            let args: Calculation = serde_json::from_value(arguments.clone())
                .map_err(|_| "Provide an expression string.".to_string())?;
            if args.expression.len() > MAX_EXPRESSION_BYTES {
                return Err("Expression must contain 1 to 1024 bytes.".into());
            }
            let expression = args.expression.trim();
            let result = calculate(expression)?;
            json!({ "expression": expression, "result": result })
        }
        TOOL_CONVERT_UNITS => {
            let args: Conversion = serde_json::from_value(arguments.clone())
                .map_err(|_| "Provide a numeric value, from_unit, and to_unit.".to_string())?;
            let result = convert(args.value, &args.from_unit, &args.to_unit)?;
            json!({ "value": args.value, "from_unit": args.from_unit,
                "to_unit": args.to_unit, "result": result })
        }
        _ => return Err("Unknown calculator operation.".into()),
    };
    Ok(result.to_string())
}

fn calculate(expression: &str) -> Result<f64, String> {
    validate_expression(expression)?;
    let parser = fasteval::Parser {
        expr_len_limit: MAX_EXPRESSION_BYTES,
        expr_depth_limit: 32,
    };
    let mut slab = fasteval::Slab::with_capacity(512);
    let parsed = parser
        .parse(expression, &mut slab.ps)
        .map_err(|error| format!("Invalid expression: {error}"))?;
    let mut invalid_function_result = false;
    let mut namespace = |name: &str, args: Vec<f64>| {
        let result = match (name, args.as_slice()) {
            ("pi", []) => Some(std::f64::consts::PI),
            ("e", []) => Some(std::f64::consts::E),
            ("sqrt", [value]) => Some(value.sqrt()),
            ("ln", [value]) => Some(value.ln()),
            ("exp", [value]) => Some(value.exp()),
            ("percent", [value]) => Some(value / 100.0),
            _ => None,
        };
        if result.is_some_and(|value| !value.is_finite()) {
            invalid_function_result = true;
        }
        result
    };
    let result = parsed
        .from(&slab.ps)
        .eval(&slab, &mut namespace)
        .map_err(|error| format!("Could not calculate: {error}"))?;
    if invalid_function_result {
        return Err("Function is outside its numeric domain or overflowed.".into());
    }
    finite(result)
}

// Restrict the parser's language to arithmetic. In particular, its print function
// must never be reachable from model-supplied input.
fn validate_expression(expression: &str) -> Result<(), String> {
    if expression.is_empty() || expression.len() > MAX_EXPRESSION_BYTES {
        return Err("Expression must contain 1 to 1024 bytes.".into());
    }
    let bytes = expression.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_digit() || byte == b'.' {
            index += 1;
            while index < bytes.len() && (bytes[index].is_ascii_digit() || bytes[index] == b'.') {
                index += 1;
            }
            if index < bytes.len() && matches!(bytes[index], b'e' | b'E') {
                index += 1;
                if index < bytes.len() && matches!(bytes[index], b'+' | b'-') {
                    index += 1;
                }
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    index += 1;
                }
            }
        } else if byte.is_ascii_alphabetic() {
            let start = index;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            if !matches!(
                &expression[start..index],
                "pi" | "e"
                    | "sqrt"
                    | "ln"
                    | "log"
                    | "exp"
                    | "percent"
                    | "abs"
                    | "min"
                    | "max"
                    | "round"
                    | "floor"
                    | "ceil"
                    | "sin"
                    | "cos"
                    | "tan"
                    | "asin"
                    | "acos"
                    | "atan"
            ) {
                return Err(format!(
                    "Unsupported function or constant: {}",
                    &expression[start..index]
                ));
            }
        } else if byte.is_ascii_whitespace() || b"+-*/^(),".contains(&byte) {
            index += 1;
        } else {
            return Err("Use arithmetic operators and supported functions only. Write percentages as percent(15) or 15/100, not %.".into());
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum Dimension {
    Length,
    Mass,
    Volume,
    Duration,
    Temperature,
}

struct Unit {
    name: &'static str,
    dimension: Dimension,
    scale: f64,
    offset: f64,
}

// SI base conversions; customary units use the international inch/pound and
// U.S. liquid measures (NIST SP 811). Explicit US names avoid imperial ambiguity.
const UNITS: &[Unit] = &[
    unit("mm", Dimension::Length, 0.001),
    unit("cm", Dimension::Length, 0.01),
    unit("m", Dimension::Length, 1.0),
    unit("km", Dimension::Length, 1000.0),
    unit("in", Dimension::Length, 0.0254),
    unit("ft", Dimension::Length, 0.3048),
    unit("yd", Dimension::Length, 0.9144),
    unit("mi", Dimension::Length, 1609.344),
    unit("mg", Dimension::Mass, 0.000001),
    unit("g", Dimension::Mass, 0.001),
    unit("kg", Dimension::Mass, 1.0),
    unit("oz", Dimension::Mass, 0.45359237 / 16.0),
    unit("lb", Dimension::Mass, 0.45359237),
    unit("ml", Dimension::Volume, 0.001),
    unit("l", Dimension::Volume, 1.0),
    unit("us_tsp", Dimension::Volume, 3.785411784 / 768.0),
    unit("us_tbsp", Dimension::Volume, 3.785411784 / 256.0),
    unit("us_fl_oz", Dimension::Volume, 3.785411784 / 128.0),
    unit("us_cup", Dimension::Volume, 3.785411784 / 16.0),
    unit("us_gal", Dimension::Volume, 3.785411784),
    unit("ms", Dimension::Duration, 0.001),
    unit("s", Dimension::Duration, 1.0),
    unit("min", Dimension::Duration, 60.0),
    unit("h", Dimension::Duration, 3600.0),
    unit("day", Dimension::Duration, 86400.0),
    Unit {
        name: "celsius",
        dimension: Dimension::Temperature,
        scale: 1.0,
        offset: 273.15,
    },
    Unit {
        name: "fahrenheit",
        dimension: Dimension::Temperature,
        scale: 5.0 / 9.0,
        offset: 273.15 - 32.0 * 5.0 / 9.0,
    },
    unit("kelvin", Dimension::Temperature, 1.0),
];

const fn unit(name: &'static str, dimension: Dimension, scale: f64) -> Unit {
    Unit {
        name,
        dimension,
        scale,
        offset: 0.0,
    }
}

pub fn unit_names() -> Vec<&'static str> {
    UNITS.iter().map(|unit| unit.name).collect()
}

fn convert(value: f64, from: &str, to: &str) -> Result<f64, String> {
    finite(value)?;
    if from.len() > 32 || to.len() > 32 {
        return Err("Unit names must use the short identifiers listed in the tool schema.".into());
    }
    let find = |name: &str| {
        UNITS.iter().find(|unit| unit.name == name).ok_or_else(|| {
            format!(
                "Unsupported unit: {name}. Use one of: {}",
                unit_names().join(", ")
            )
        })
    };
    let from = find(from)?;
    let to = find(to)?;
    if from.dimension != to.dimension {
        return Err(
            "Cannot convert between different kinds of quantity (for example mass and volume)."
                .into(),
        );
    }
    if from.dimension == Dimension::Temperature {
        let absolute_zero = match from.name {
            "celsius" => -273.15,
            "fahrenheit" => -459.67,
            _ => 0.0,
        };
        if value < absolute_zero {
            return Err("Temperature cannot be below absolute zero.".into());
        }
    }
    if from.name == to.name {
        return Ok(value);
    }
    let mut base = finite(value * from.scale + from.offset)?;
    if from.dimension == Dimension::Temperature {
        // The exact absolute-zero inputs can round a few ulps below zero.
        base = base.max(0.0);
    }
    finite((base - to.offset) / to.scale)
}

fn finite(value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(if value == 0.0 { 0.0 } else { value })
    } else {
        Err(
            "Result is not finite. Check division by zero, function domains, and numeric overflow."
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn arithmetic_and_functions() {
        for (expression, expected) in [
            ("2 + 3 * 4", 14.0),
            ("(2 + 3) * 4", 20.0),
            ("8 / 2 * 2", 8.0),
            ("10 - 3 - 2", 5.0),
            ("2^3^2", 512.0),
            ("-2^2", 4.0),
            ("-(2^2)", -4.0),
            ("(-2)^2", 4.0),
            ("2^-3", 0.125),
            ("10 / 2 / 5", 1.0),
            ("1/4", 0.25),
            ("250 * percent(15)", 37.5),
            ("1.2e3 + 1E-2", 1200.01),
            ("sqrt(81) + abs(-2)", 11.0),
            ("ln(exp(2))", 2.0),
            ("sin(pi / 2)", 1.0),
            ("min(3, 1, 2) + max(4, 5)", 6.0),
            ("log(100)", 2.0),
            ("round(1.6) + floor(1.6) + ceil(1.6)", 5.0),
        ] {
            near(calculate(expression).expect(expression), expected);
        }
    }

    #[test]
    fn rejects_unsafe_invalid_and_unbounded_input() {
        for expression in [
            "",
            "print(1)",
            "1 + print(2)",
            "x=4",
            "1;2",
            "1<2",
            "\"text\"",
            "15%",
            "1/0",
            "sqrt(-1)",
            "min(sqrt(-1), 2)",
            "ln(0)",
            "10^1000",
            "1+",
            "sqrt(1,2)",
            "1K",
        ] {
            assert!(calculate(expression).is_err(), "accepted {expression}");
        }
        assert!(calculate(&"1".repeat(1025)).is_err());
        assert!(calculate(&format!("{}1{}", "(".repeat(40), ")".repeat(40))).is_err());
        assert!(calculate(&format!("{}1", "-".repeat(100))).is_err());
    }

    #[test]
    fn conversions_and_round_trips() {
        for (value, from, to, expected) in [
            (1.0, "in", "cm", 2.54),
            (1.0, "lb", "kg", 0.45359237),
            (1.0, "us_gal", "l", 3.785411784),
            (1.0, "us_cup", "us_tbsp", 16.0),
            (2.0, "h", "min", 120.0),
            (32.0, "fahrenheit", "celsius", 0.0),
            (100.0, "celsius", "fahrenheit", 212.0),
            (-40.0, "celsius", "fahrenheit", -40.0),
            (0.0, "kelvin", "celsius", -273.15),
            (-459.67, "fahrenheit", "kelvin", 0.0),
        ] {
            near(convert(value, from, to).unwrap(), expected);
        }
        for from in UNITS {
            for to in UNITS.iter().filter(|to| to.dimension == from.dimension) {
                let converted = convert(123.45, from.name, to.name).unwrap();
                near(convert(converted, to.name, from.name).unwrap(), 123.45);
            }
        }
        for (value, from, to) in [
            (1.0, "kg", "l"),
            (1.0, "gal", "l"),
            (-1.0, "kelvin", "celsius"),
            (-1e-12, "kelvin", "kelvin"),
            (-274.0, "celsius", "fahrenheit"),
            (f64::MAX, "km", "mm"),
        ] {
            assert!(convert(value, from, to).is_err());
        }
    }
}
