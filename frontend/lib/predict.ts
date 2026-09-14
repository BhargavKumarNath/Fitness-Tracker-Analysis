export function predictActivityBaseline(steps: number, heartRate: number): string {
  if (heartRate >= 150 || steps >= 15000) return "running";
  if (heartRate >= 125 || steps >= 8000) return "cycling";
  if (steps === 0 && heartRate < 90) return "yoga";
  return "walking";
}

export function predictCaloriesBaseline(steps: number, heartRate: number, sleepHours: number, activityType: string): number {
  const activityFactor: Record<string, number> = { walking: 1, running: 1.35, cycling: 1.2, swimming: 1.1, yoga: 0.75, hiking: 1.45, gym_workout: 1.15 };
  const factor = activityFactor[activityType] ?? 1;
  const sleepAdjustment = Math.max(0.85, Math.min(1.1, 1 + (7.5 - sleepHours) * 0.02));
  return Math.max(50, (80 + steps * 0.035 + Math.max(0, heartRate - 60) * 1.5) * factor * sleepAdjustment);
}