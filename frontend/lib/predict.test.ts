import { describe, expect, it } from "vitest";
import { predictActivityBaseline, predictCaloriesBaseline } from "./predict";

describe("baseline predictions", () => {
  it("matches the Python activity thresholds", () => {
    expect(predictActivityBaseline(0, 68)).toBe("yoga");
    expect(predictActivityBaseline(16000, 120)).toBe("running");
    expect(predictActivityBaseline(8000, 100)).toBe("cycling");
    expect(predictActivityBaseline(4000, 100)).toBe("walking");
  });

  it("matches the Python calorie formula", () => {
    expect(predictCaloriesBaseline(8000, 130, 7.5, "hiking")).toBeCloseTo(674.25, 10);
    expect(predictCaloriesBaseline(0, 68, 8, "yoga")).toBeGreaterThanOrEqual(50);
    expect(predictCaloriesBaseline(8000, 130, 7.5, "Yoga")).toBe(predictCaloriesBaseline(8000, 130, 7.5, "unknown"));
  });
});