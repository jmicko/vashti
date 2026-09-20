import assert from "node:assert/strict";
import test from "node:test";
import { calculatorResult } from "./calculatorResult.ts";

test("calculator results display expressions and converted units", () => {
  assert.deepEqual(calculatorResult({ name: "calculate", result: JSON.stringify({ expression: "250 * percent(15)", result: 37.5 }) }),
    { expression: "250 * percent(15)", result: "37.5" });
  assert.deepEqual(calculatorResult({ name: "convert_units", result: JSON.stringify({ value: 32, from_unit: "fahrenheit", to_unit: "celsius", result: 0 }) }),
    { expression: "32 fahrenheit", result: "0 celsius" });
});

test("malformed and unsuccessful results keep the generic inspector", () => {
  for (const result of ["broken", "null", "{}", '{"result":1e999}', '{"error":"bad input","result":1}', '{"expression":"1","result":"1"}']) {
    assert.equal(calculatorResult({ name: "calculate", result }), null);
  }
  assert.equal(calculatorResult({ name: "search_notes", result: '{"expression":"1","result":1}' }), null);
});

test("compact results suppress floating-point noise without losing small values", () => {
  for (const [value, expected] of [[211.99999999999997, "212"], [0.30000000000000004, "0.3"], [1e-20, "1e-20"]]) {
    assert.equal(calculatorResult({ name: "calculate", result: JSON.stringify({ expression: "test", result: value }) }).result, expected);
  }
});
